#!/usr/bin/env python3
"""
Execution Filter for Code Quality

Filters training data by actually RUNNING the code.
- Parses Python syntax
- Executes with timeout
- Validates test assertions
- Rejects anything that crashes

Usage:
    python -m foundry.execution_filter --input data/raw.jsonl --output data/verified.jsonl
"""

import ast
import sys
import json
import argparse
import multiprocessing
import traceback
from io import StringIO
from typing import Optional, Tuple
from contextlib import redirect_stdout, redirect_stderr
import signal

# Timeout for code execution (seconds)
TIMEOUT = 5

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")

def extract_python_code(text: str) -> list[str]:
    """Extract Python code blocks from text."""
    code_blocks = []
    
    # Try to find ```python blocks
    if "```python" in text:
        parts = text.split("```python")
        for part in parts[1:]:
            if "```" in part:
                code = part.split("```")[0].strip()
                code_blocks.append(code)
    
    # Try to find ``` blocks (generic)
    elif "```" in text:
        parts = text.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Odd indices are code blocks
                code = part.strip()
                if code and not code.startswith(('bash', 'shell', 'json', 'yaml', 'sql')):
                    code_blocks.append(code)
    
    # If no code blocks, try to find indented code or def/class
    if not code_blocks:
        lines = text.split('\n')
        in_code = False
        current_block = []
        
        for line in lines:
            if line.strip().startswith(('def ', 'class ', 'import ', 'from ')):
                in_code = True
            if in_code:
                if line.strip() == '' and current_block and not current_block[-1].strip().endswith(':'):
                    # End of code block
                    code_blocks.append('\n'.join(current_block))
                    current_block = []
                    in_code = False
                else:
                    current_block.append(line)
        
        if current_block:
            code_blocks.append('\n'.join(current_block))
    
    return code_blocks


def check_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Check if code has valid Python syntax."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno})"


def execute_code(code: str, timeout: int = TIMEOUT) -> Tuple[bool, str, str]:
    """
    Execute code in isolated environment with timeout.
    Returns: (success, stdout, stderr)
    """
    # Capture stdout/stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    
    # Create restricted globals (no file/network access)
    restricted_globals = {
        '__builtins__': {
            'print': print,
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sorted': sorted,
            'reversed': reversed,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'None': None,
            'True': True,
            'False': False,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'isinstance': isinstance,
            'hasattr': hasattr,
            'getattr': getattr,
            'setattr': setattr,
            'type': type,
            'callable': callable,
            'iter': iter,
            'next': next,
            'Exception': Exception,
            'ValueError': ValueError,
            'TypeError': TypeError,
            'KeyError': KeyError,
            'IndexError': IndexError,
            'AssertionError': AssertionError,
            'StopIteration': StopIteration,
        }
    }
    
    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, restricted_globals, {})
        signal.alarm(0)  # Cancel timeout
        return True, stdout_capture.getvalue(), stderr_capture.getvalue()
    
    except TimeoutError:
        return False, "", "TimeoutError: Execution exceeded time limit"
    except AssertionError as e:
        return False, stdout_capture.getvalue(), f"AssertionError: {e}"
    except Exception as e:
        return False, stdout_capture.getvalue(), f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)


def validate_sample(sample: dict) -> Tuple[bool, dict, str]:
    """
    Validate a training sample by executing its code.
    
    Returns: (is_valid, sample, reason)
    """
    text = sample.get('response', '') or sample.get('text', '') or sample.get('code', '')
    
    if not text:
        return False, sample, "No code content found"
    
    code_blocks = extract_python_code(text)
    
    if not code_blocks:
        # No code blocks - might be conceptual, keep it
        return True, sample, "No code to validate (conceptual)"
    
    # Validate each code block
    all_valid = True
    reasons = []
    
    for i, code in enumerate(code_blocks):
        # Check syntax
        syntax_ok, syntax_err = check_syntax(code)
        if not syntax_ok:
            all_valid = False
            reasons.append(f"Block {i+1}: {syntax_err}")
            continue
        
        # Try to execute (only if it looks like it should run)
        # Skip if it's just a class/function definition without tests
        has_execution = any(x in code for x in ['print(', 'assert ', 'if __name__', '()'])
        
        if has_execution:
            success, stdout, stderr = execute_code(code)
            if not success:
                all_valid = False
                reasons.append(f"Block {i+1}: {stderr}")
            else:
                reasons.append(f"Block {i+1}: Executed OK")
        else:
            reasons.append(f"Block {i+1}: Syntax OK (definition only)")
    
    reason = "; ".join(reasons)
    return all_valid, sample, reason


def process_file(input_path: str, output_path: str, reject_path: str = None):
    """Process JSONL file, filtering by execution."""
    
    valid_count = 0
    invalid_count = 0
    total_count = 0
    
    reject_file = open(reject_path, 'w') if reject_path else None
    
    print(f"Processing {input_path}...")
    print(f"Output: {output_path}")
    if reject_path:
        print(f"Rejects: {reject_path}")
    print()
    
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            total_count += 1
            
            try:
                sample = json.loads(line.strip())
            except json.JSONDecodeError:
                invalid_count += 1
                continue
            
            is_valid, sample, reason = validate_sample(sample)
            
            if is_valid:
                valid_count += 1
                fout.write(json.dumps(sample) + '\n')
            else:
                invalid_count += 1
                if reject_file:
                    sample['_reject_reason'] = reason
                    reject_file.write(json.dumps(sample) + '\n')
            
            if total_count % 100 == 0:
                pct = valid_count / total_count * 100
                print(f"  [{total_count}] Valid: {valid_count} ({pct:.1f}%) | Invalid: {invalid_count}")
    
    if reject_file:
        reject_file.close()
    
    print()
    print("="*50)
    print(f"FILTERING COMPLETE")
    print("="*50)
    print(f"Total samples:    {total_count}")
    print(f"Valid (kept):     {valid_count} ({valid_count/total_count*100:.1f}%)")
    print(f"Invalid (dropped): {invalid_count} ({invalid_count/total_count*100:.1f}%)")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Filter code samples by execution')
    parser.add_argument('--input', '-i', required=True, help='Input JSONL file')
    parser.add_argument('--output', '-o', required=True, help='Output JSONL file (valid samples)')
    parser.add_argument('--rejects', '-r', help='Output JSONL file for rejected samples')
    parser.add_argument('--timeout', '-t', type=int, default=5, help='Execution timeout (seconds)')
    args = parser.parse_args()
    
    global TIMEOUT
    TIMEOUT = args.timeout
    
    process_file(args.input, args.output, args.rejects)


if __name__ == '__main__':
    main()
