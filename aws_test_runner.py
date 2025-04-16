#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
import time
import json
from datetime import datetime

def setup_environment():
    """Install dependencies needed for testing"""
    print("Setting up test environment...")
    
    # Create a requirements file for the test dependencies
    with open('test_requirements.txt', 'w') as f:
        f.write("aiohttp==3.8.4\n")
        f.write("asyncio==3.4.3\n")
        f.write("matplotlib==3.7.1\n")
        f.write("numpy==1.24.3\n")
        f.write("sqlalchemy==2.0.15\n")
        f.write("python-dotenv==1.0.0\n")
        f.write("statistics==1.0.3.5\n")
    
    # Install the dependencies
    subprocess.run(["pip", "install", "-r", "test_requirements.txt"])
    print("Test environment setup complete.")

def run_api_stress_test(api_url, username, password, queries, concurrency, delay):
    """Run the API stress test"""
    print(f"Running API stress test against {api_url} with {queries} queries and {concurrency} concurrent users...")
    
    command = [
        "python", "stress_test.py",
        "--url", api_url,
        "--username", username,
        "--password", password,
        "--queries", str(queries),
        "--concurrency", str(concurrency),
        "--delay", str(delay)
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Create a directory for test results if it doesn't exist
    os.makedirs("test_results", exist_ok=True)
    
    # Save the output to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"test_results/api_test_output_{timestamp}.txt"
    
    with open(output_file, 'w') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n\nERRORS:\n")
            f.write(result.stderr)
    
    print(f"API test completed. Results saved to {output_file}")
    
    # Return True if the test was successful
    return result.returncode == 0

def run_db_performance_test(db_url, iterations):
    """Run the database performance test"""
    print(f"Running database performance test with {iterations} iterations...")
    
    command = [
        "python", "db_performance_test.py",
        "--iterations", str(iterations)
    ]
    
    if db_url:
        command.extend(["--db", db_url])
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Create a directory for test results if it doesn't exist
    os.makedirs("test_results", exist_ok=True)
    
    # Save the output to a file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"test_results/db_test_output_{timestamp}.txt"
    
    with open(output_file, 'w') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n\nERRORS:\n")
            f.write(result.stderr)
    
    print(f"Database test completed. Results saved to {output_file}")
    
    # Return True if the test was successful
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='Run stress and performance tests on AWS')
    
    # Common arguments
    parser.add_argument('--setup', action='store_true', help='Setup the test environment')
    
    # API stress test arguments
    parser.add_argument('--api-test', action='store_true', help='Run the API stress test')
    parser.add_argument('--api-url', type=str, help='URL of the API to test')
    parser.add_argument('--username', type=str, help='Username for API authentication')
    parser.add_argument('--password', type=str, help='Password for API authentication')
    parser.add_argument('--queries', type=int, default=1000, help='Number of queries to send')
    parser.add_argument('--concurrency', type=int, default=10, help='Number of concurrent requests')
    parser.add_argument('--delay', type=int, default=100, help='Delay between requests in ms')
    
    # Database test arguments
    parser.add_argument('--db-test', action='store_true', help='Run the database performance test')
    parser.add_argument('--db-url', type=str, help='Database URL for testing')
    parser.add_argument('--iterations', type=int, default=100, help='Number of test iterations')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.setup:
        setup_environment()
    
    if args.api_test:
        if not args.api_url:
            print("Error: --api-url is required for API testing")
            sys.exit(1)
        
        api_success = run_api_stress_test(
            args.api_url,
            args.username or os.getenv("TEST_USERNAME", "admin"),
            args.password or os.getenv("TEST_PASSWORD", "adminpassword"),
            args.queries,
            args.concurrency,
            args.delay
        )
        
        if not api_success:
            print("API stress test failed!")
    
    if args.db_test:
        db_success = run_db_performance_test(
            args.db_url,
            args.iterations
        )
        
        if not db_success:
            print("Database performance test failed!")

if __name__ == "__main__":
    main()
