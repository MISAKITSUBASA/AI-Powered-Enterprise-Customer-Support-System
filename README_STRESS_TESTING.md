# Stress Testing Guide for AI Customer Support System

This guide explains how to use the included stress testing tools to evaluate the performance of your deployed AI Customer Support System under high load conditions.

## Prerequisites

Before running the stress tests, ensure you have:

1. A running instance of the AI Customer Support System (local or deployed on AWS)
2. Python 3.8+ installed on your testing machine
3. The required Python packages:
   ```
   pip install aiohttp asyncio matplotlib numpy sqlalchemy python-dotenv
   ```

## API Stress Testingsting

The `stress_test.py` script simulates multiple concurrent users sending queries to the chat API endpoint.
Running locally is useful for:
### Usage development and debugging
- Testing basic functionality
```bash iterations during development
python stress_test.py --url [API_URL] --username [USERNAME] --password [PASSWORD] --queries [NUM_QUERIES] --concurrency [CONCURRENT_REQUESTS] --delay [MS_DELAY]
```mple local test command:
```bash
Parameters:ss_test.py --url http://localhost:8000 --queries 500 --concurrency 5
- `--url`: Base URL of your API (default: http://localhost:8000)
- `--username`: Admin username for authentication
- `--password`: Password for authentication
- `--queries`: Total number of queries to send (default: 1000)
- `--concurrency`: Number of concurrent requests (default: 10)
- `--delay`: Delay between requests in milliseconds (default: 100)
- Full-scale stress testing
### Example
Example remote test command:
To run a stress test against your AWS deployment with 2000 requests and 25 concurrent users:
python stress_test.py --url https://your-aws-api-url.com --queries 2000 --concurrency 25
```bash
python stress_test.py --url https://your-aws-api-url.com --queries 2000 --concurrency 25
``` best results, run the test scripts from a different machine than your API server to avoid resource contention affecting results.

## Database Performance Testing

The `db_performance_test.py` script tests database query performance across different operations.ndpoint.

### Usage

```bash
python db_performance_test.py --db [DATABASE_URL] --iterations [NUM_ITERATIONS]D] --queries [NUM_QUERIES] --concurrency [CONCURRENT_REQUESTS] --delay [MS_DELAY]
```

Parameters:
- `--db`: Database URL (defaults to the DATABASE_URL in your .env file)
- `--iterations`: Number of test iterations for each query (default: 100)
- `--password`: Password for authentication
### Examples`: Total number of queries to send (default: 1000)
- `--concurrency`: Number of concurrent requests (default: 10)
```bashlay`: Delay between requests in milliseconds (default: 100)
python db_performance_test.py --iterations 200
``` Example

## Running Tests on AWS EC2your AWS deployment with 2000 requests and 25 concurrent users:

For the most realistic stress testing, you can run the tests from an EC2 instance:
python stress_test.py --url https://your-aws-api-url.com --queries 2000 --concurrency 25
### Setting Up a Test EC2 Instance

1. Launch a t3.medium (or larger) EC2 instance with Amazon Linux 2
2. Install required software:
   ```bashThe `db_performance_test.py` script tests database query performance across different operations.
   sudo yum update -y
   sudo yum install -y git python3 python3-pip### Usage
   ```

3. Clone your repository or upload test scripts: [NUM_ITERATIONS]
   ```bash
   git clone https://github.com/yourusername/AI-Powered-Enterprise-Customer-Support-System.git
   cd AI-Powered-Enterprise-Customer-Support-SystemParameters:
   ``` to the DATABASE_URL in your .env file)
- `--iterations`: Number of test iterations for each query (default: 100)
4. Or upload your test scripts directly using SCP:
   ```bash### Example
   scp -i your-key.pem stress_test.py db_performance_test.py aws_test_runner.py ec2-user@your-ec2-instance:~/
   ```
200
### Running Tests with the Helper Script

The included `aws_test_runner.py` script simplifies running tests on EC2:

```bash
# Setup the environment first
python aws_test_runner.py --setup
- A summary in the console showing request success rates, response times, and throughput
# Run API stress testults from each individual request
python aws_test_runner.py --api-test --api-url https://your-aws-api-url.com --queries 2000 --concurrency 25

# Run database performance test (if your EC2 has access to your DB)
python aws_test_runner.py --db-test --db-url postgresql://user:pass@your-rds-instance:5432/dbname --iterations 200
```The database test generates:
erent query types (select, complex analytics, etc.)
### Testing from Multiple Regions- A box plot showing the distribution of query execution times
ions
For realistic global performance testing, launch EC2 instances in different AWS regions and run tests simultaneously:- The plots are saved as PNG files for documentation

1. Launch similar EC2 instances in regions like:
   - us-east-1 (N. Virginia)
   - eu-west-1 (Ireland)
   - ap-southeast-1 (Singapore)
1. **API Performance Improvements**:
2. Run the same test script on each instance and compare results.er resources (CPU/memory) on AWS
   - Implement caching for frequent queries
### Analyzing Cross-Region Resultsoad balancing

To consolidate results from multiple regions:
columns
1. Collect the JSON result files from each region results
2. Use the following command format to run tests with region labels:
   ```bash
   python stress_test.py --url https://your-aws-api-url.com --queries 1000 --concurrency 10 --region us-east-1
   ```
   - Implement pagination for large result sets






























































For detailed logs, check your application's log files during the stress test.5. Look for database connection limits or API rate limiting4. Check AWS security groups and network access if testing against deployed instances3. Ensure your database connection string is correct2. Verify the API endpoint is accessible1. Check your authentication credentialsIf you encounter errors during testing:## Troubleshooting- Less than 5% error rate under full load- Support for 100+ requests per minute- Ability to handle 10-20 concurrent users with minimal degradation- Database query times under 50ms for simple queries- API response times under 500ms for chat requests (95th percentile)For a well-optimized system running on moderate AWS resources:## Typical Performance Benchmarks   - Consider using a smaller or more efficient language model   - Optimize your LLM prompt templates   - Use database connection pooling   - Implement pagination for large result sets3. **Reducing Response Times**:   - Migrate from SQLite to PostgreSQL for production loads   - Consider database sharding for larger deployments   - Optimize complex queries identified in the test results   - Add indexes for frequently queried columns2. **Database Optimizations**:   - Consider horizontal scaling with load balancing   - Implement caching for frequent queries   - Increase server resources (CPU/memory) on AWS1. **API Performance Improvements**:If performance issues are identified:## Performance Optimization Tips- The plots are saved as PNG files for documentation- A bar chart with average execution times and standard deviations- A box plot showing the distribution of query execution times- Performance statistics for different query types (select, complex analytics, etc.)The database test generates:### Database Performance Results- Metrics on escalated requests and error rates- A detailed JSON file with results from each individual request- A summary in the console showing request success rates, response times, and throughputThe stress test generates:### API Stress Test Results## Understanding Test Results3. Compare response times and error rates between regions to identify potential performance issues for global users   - Use database connection pooling
   - Optimize your LLM prompt templates
   - Consider using a smaller or more efficient language model

## Typical Performance Benchmarks

For a well-optimized system running on moderate AWS resources:

- API response times under 500ms for chat requests (95th percentile)
- Database query times under 50ms for simple queries
- Ability to handle 10-20 concurrent users with minimal degradation
- Support for 100+ requests per minute
- Less than 5% error rate under full load

## Troubleshooting

If you encounter errors during testing:

1. Check your authentication credentials
2. Verify the API endpoint is accessible
3. Ensure your database connection string is correct
4. Check AWS security groups and network access if testing against deployed instances
5. Look for database connection limits or API rate limiting

For detailed logs, check your application's log files during the stress test.
