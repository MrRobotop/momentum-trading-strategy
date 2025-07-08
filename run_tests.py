#!/usr/bin/env python3
"""
Comprehensive Test Runner for Momentum Strategy
===============================================

Automated test runner that executes all test suites including:
- Python unit tests
- Integration tests  
- Performance benchmarks
- React component tests
- API endpoint tests
- Security checks

Author: Rishabh Ashok Patil
"""

import os
import sys
import subprocess
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
import concurrent.futures
from typing import Dict, List, Tuple

class TestRunner:
    """Comprehensive test runner for the momentum strategy project"""
    
    def __init__(self, verbose: bool = False, parallel: bool = True):
        self.verbose = verbose
        self.parallel = parallel
        self.project_root = Path(__file__).parent
        self.results = {}
        self.start_time = time.time()
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {
            "INFO": "ℹ️",
            "SUCCESS": "✅", 
            "ERROR": "❌",
            "WARNING": "⚠️",
            "RUNNING": "🏃"
        }.get(level, "📝")
        
        print(f"[{timestamp}] {prefix} {message}")
        
        if self.verbose and level == "ERROR":
            print(f"    └─ Check logs for detailed error information")
    
    def run_command(self, command: List[str], cwd: Path = None, 
                   timeout: int = 300) -> Tuple[bool, str, str]:
        """Run a command and return success status, stdout, stderr"""
        try:
            if cwd is None:
                cwd = self.project_root
                
            self.log(f"Running: {' '.join(command)}", "RUNNING")
            
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            success = result.returncode == 0
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return False, "", str(e)
    
    def check_prerequisites(self) -> bool:
        """Check if all required tools are installed"""
        self.log("Checking prerequisites...", "RUNNING")
        
        required_tools = [
            ("python", ["python", "--version"]),
            ("pip", ["pip", "--version"]),
            ("node", ["node", "--version"]),
            ("npm", ["npm", "--version"])
        ]
        
        missing_tools = []
        
        for tool_name, command in required_tools:
            success, stdout, stderr = self.run_command(command, timeout=10)
            if success:
                version = stdout.strip().split('\n')[0]
                self.log(f"{tool_name}: {version}")
            else:
                missing_tools.append(tool_name)
                self.log(f"{tool_name}: Not found", "ERROR")
        
        if missing_tools:
            self.log(f"Missing required tools: {', '.join(missing_tools)}", "ERROR")
            return False
        
        self.log("All prerequisites satisfied", "SUCCESS")
        return True
    
    def setup_python_environment(self) -> bool:
        """Set up Python virtual environment and install dependencies"""
        self.log("Setting up Python environment...", "RUNNING")
        
        # Check if virtual environment exists
        venv_path = self.project_root / "venv"
        if not venv_path.exists():
            self.log("Creating virtual environment...")
            success, _, stderr = self.run_command([
                "python", "-m", "venv", "venv"
            ])
            if not success:
                self.log(f"Failed to create virtual environment: {stderr}", "ERROR")
                return False
        
        # Install dependencies
        pip_command = ["venv/bin/pip"] if os.name != 'nt' else ["venv\\Scripts\\pip"]
        
        success, _, stderr = self.run_command([
            *pip_command, "install", "-r", "requirements.txt"
        ])
        
        if not success:
            self.log(f"Failed to install Python dependencies: {stderr}", "ERROR")
            return False
        
        # Install test dependencies
        test_deps = [
            "pytest", "pytest-cov", "pytest-benchmark", 
            "flake8", "black", "isort", "bandit", "safety"
        ]
        
        success, _, stderr = self.run_command([
            *pip_command, "install", *test_deps
        ])
        
        if not success:
            self.log(f"Failed to install test dependencies: {stderr}", "ERROR")
            return False
        
        self.log("Python environment ready", "SUCCESS")
        return True
    
    def setup_node_environment(self) -> bool:
        """Set up Node.js environment and install dependencies"""
        self.log("Setting up Node.js environment...", "RUNNING")
        
        dashboard_path = self.project_root / "dashboard"
        
        # Install npm dependencies
        success, _, stderr = self.run_command([
            "npm", "install"
        ], cwd=dashboard_path)
        
        if not success:
            self.log(f"Failed to install Node dependencies: {stderr}", "ERROR")
            return False
        
        self.log("Node.js environment ready", "SUCCESS")
        return True
    
    def run_python_tests(self) -> Dict:
        """Run Python unit tests"""
        self.log("Running Python unit tests...", "RUNNING")
        
        python_command = ["venv/bin/python"] if os.name != 'nt' else ["venv\\Scripts\\python"]
        
        # Run tests with coverage
        success, stdout, stderr = self.run_command([
            *python_command, "-m", "pytest", "tests/test_strategy.py", 
            "-v", "--cov=momentum_strategy", "--cov-report=html", "--cov-report=term"
        ])
        
        result = {
            "name": "Python Unit Tests",
            "success": success,
            "output": stdout,
            "error": stderr,
            "duration": 0  # Will be calculated by caller
        }
        
        if success:
            self.log("Python tests passed", "SUCCESS")
        else:
            self.log("Python tests failed", "ERROR")
            if self.verbose:
                print(f"STDOUT:\n{stdout}")
                print(f"STDERR:\n{stderr}")
        
        return result
    
    def run_react_tests(self) -> Dict:
        """Run React component tests"""
        self.log("Running React component tests...", "RUNNING")
        
        dashboard_path = self.project_root / "dashboard"
        
        success, stdout, stderr = self.run_command([
            "npm", "test", "--", "--coverage", "--watchAll=false"
        ], cwd=dashboard_path)
        
        result = {
            "name": "React Component Tests",
            "success": success,
            "output": stdout,
            "error": stderr,
            "duration": 0
        }
        
        if success:
            self.log("React tests passed", "SUCCESS")
        else:
            self.log("React tests failed", "ERROR")
            if self.verbose:
                print(f"STDOUT:\n{stdout}")
                print(f"STDERR:\n{stderr}")
        
        return result
    
    def run_linting_checks(self) -> Dict:
        """Run code linting and formatting checks"""
        self.log("Running linting checks...", "RUNNING")
        
        python_command = ["venv/bin/python"] if os.name != 'nt' else ["venv\\Scripts\\python"]
        
        checks = [
            (["venv/bin/flake8" if os.name != 'nt' else "venv\\Scripts\\flake8", 
              "momentum_strategy/", "--max-line-length=100"], "Flake8"),
            (["venv/bin/black" if os.name != 'nt' else "venv\\Scripts\\black", 
              "--check", "momentum_strategy/"], "Black formatting"),
            (["venv/bin/isort" if os.name != 'nt' else "venv\\Scripts\\isort", 
              "--check-only", "momentum_strategy/"], "Import sorting")
        ]
        
        all_passed = True
        outputs = []
        
        for command, check_name in checks:
            success, stdout, stderr = self.run_command(command)
            outputs.append(f"{check_name}: {'PASSED' if success else 'FAILED'}")
            if not success:
                all_passed = False
                outputs.append(f"  Error: {stderr}")
        
        result = {
            "name": "Linting Checks",
            "success": all_passed,
            "output": "\n".join(outputs),
            "error": "",
            "duration": 0
        }
        
        if all_passed:
            self.log("All linting checks passed", "SUCCESS")
        else:
            self.log("Some linting checks failed", "ERROR")
        
        return result
    
    def run_security_checks(self) -> Dict:
        """Run security vulnerability checks"""
        self.log("Running security checks...", "RUNNING")
        
        # Bandit security check
        success1, stdout1, stderr1 = self.run_command([
            "venv/bin/bandit" if os.name != 'nt' else "venv\\Scripts\\bandit",
            "-r", "momentum_strategy/", "-f", "txt"
        ])
        
        # Safety dependency check
        success2, stdout2, stderr2 = self.run_command([
            "venv/bin/safety" if os.name != 'nt' else "venv\\Scripts\\safety",
            "check"
        ])
        
        all_passed = success1 and success2
        output = f"Bandit: {'PASSED' if success1 else 'FAILED'}\n"
        output += f"Safety: {'PASSED' if success2 else 'FAILED'}\n"
        
        if not success1:
            output += f"Bandit output:\n{stdout1}\n"
        if not success2:
            output += f"Safety output:\n{stdout2}\n"
        
        result = {
            "name": "Security Checks",
            "success": all_passed,
            "output": output,
            "error": stderr1 + stderr2,
            "duration": 0
        }
        
        if all_passed:
            self.log("Security checks passed", "SUCCESS")
        else:
            self.log("Security issues found", "WARNING")
        
        return result
    
    def run_performance_tests(self) -> Dict:
        """Run performance benchmarks"""
        self.log("Running performance benchmarks...", "RUNNING")
        
        python_command = ["venv/bin/python"] if os.name != 'nt' else ["venv\\Scripts\\python"]
        
        # Run the strategy with timing
        start_time = time.time()
        success, stdout, stderr = self.run_command([
            *python_command, "main.py", "--universe", "global_equity", 
            "--start", "2023-01-01", "--end", "2023-06-01"
        ], cwd=self.project_root / "momentum_strategy")
        
        duration = time.time() - start_time
        
        result = {
            "name": "Performance Benchmarks",
            "success": success,
            "output": f"Strategy execution time: {duration:.2f} seconds\n{stdout}",
            "error": stderr,
            "duration": duration
        }
        
        if success:
            self.log(f"Performance test completed in {duration:.2f}s", "SUCCESS")
        else:
            self.log("Performance test failed", "ERROR")
        
        return result
    
    def run_all_tests(self) -> Dict:
        """Run all test suites"""
        self.log("Starting comprehensive test suite...", "RUNNING")
        self.log(f"Author: Rishabh Ashok Patil", "INFO")
        self.log(f"Timestamp: {datetime.now().isoformat()}", "INFO")
        
        # Check prerequisites
        if not self.check_prerequisites():
            return {"success": False, "error": "Prerequisites not met"}
        
        # Set up environments
        if not self.setup_python_environment():
            return {"success": False, "error": "Python environment setup failed"}
        
        if not self.setup_node_environment():
            return {"success": False, "error": "Node environment setup failed"}
        
        # Define test suites
        test_suites = [
            ("python_tests", self.run_python_tests),
            ("react_tests", self.run_react_tests),
            ("linting", self.run_linting_checks),
            ("security", self.run_security_checks),
            ("performance", self.run_performance_tests)
        ]
        
        results = {}
        
        if self.parallel:
            # Run tests in parallel where possible
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = {}
                
                for suite_name, test_func in test_suites:
                    if suite_name in ["python_tests", "linting", "security"]:
                        # These can run in parallel
                        future = executor.submit(self._run_timed_test, test_func)
                        futures[future] = suite_name
                
                # Wait for parallel tests
                for future in concurrent.futures.as_completed(futures):
                    suite_name = futures[future]
                    results[suite_name] = future.result()
                
                # Run remaining tests sequentially
                for suite_name, test_func in test_suites:
                    if suite_name not in results:
                        results[suite_name] = self._run_timed_test(test_func)
        else:
            # Run tests sequentially
            for suite_name, test_func in test_suites:
                results[suite_name] = self._run_timed_test(test_func)
        
        # Calculate overall results
        total_duration = time.time() - self.start_time
        all_passed = all(result["success"] for result in results.values())
        
        summary = {
            "success": all_passed,
            "total_duration": total_duration,
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "author": "Rishabh Ashok Patil"
        }
        
        self.log(f"Test suite completed in {total_duration:.2f}s", 
                "SUCCESS" if all_passed else "ERROR")
        
        return summary
    
    def _run_timed_test(self, test_func) -> Dict:
        """Run a test function with timing"""
        start_time = time.time()
        result = test_func()
        result["duration"] = time.time() - start_time
        return result
    
    def generate_report(self, results: Dict) -> str:
        """Generate a comprehensive test report"""
        report = []
        report.append("=" * 60)
        report.append("MOMENTUM STRATEGY TEST REPORT")
        report.append("=" * 60)
        report.append(f"Author: {results.get('author', 'Unknown')}")
        report.append(f"Timestamp: {results.get('timestamp', 'Unknown')}")
        report.append(f"Total Duration: {results.get('total_duration', 0):.2f} seconds")
        report.append(f"Overall Status: {'PASSED' if results.get('success') else 'FAILED'}")
        report.append("")
        
        # Individual test results
        for suite_name, suite_result in results.get("results", {}).items():
            status = "PASSED" if suite_result["success"] else "FAILED"
            duration = suite_result.get("duration", 0)
            
            report.append(f"{suite_result['name']}: {status} ({duration:.2f}s)")
            
            if not suite_result["success"] and suite_result.get("error"):
                report.append(f"  Error: {suite_result['error'][:200]}...")
            
            report.append("")
        
        return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description="Run comprehensive test suite")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose output")
    parser.add_argument("--sequential", action="store_true",
                       help="Run tests sequentially instead of parallel")
    parser.add_argument("--output", "-o", help="Output file for test report")
    
    args = parser.parse_args()
    
    runner = TestRunner(verbose=args.verbose, parallel=not args.sequential)
    results = runner.run_all_tests()
    
    # Generate and display report
    report = runner.generate_report(results)
    print("\n" + report)
    
    # Save report to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if results.get("success") else 1)

if __name__ == "__main__":
    main()
