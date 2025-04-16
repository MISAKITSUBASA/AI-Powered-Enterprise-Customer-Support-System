import asyncio
import aiohttp
import time
import random
import argparse
import json
import statistics
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Sample questions to use for testing
SAMPLE_QUESTIONS = [
    "How do I reset my password?",
    "What are your business hours?",
    "Can I return a product after 30 days?",
    "How do I track my order?",
    "I need to update my shipping address",
    "Do you ship internationally?",
    "How much is the shipping cost?",
    "Can I cancel my subscription?",
    "What payment methods do you accept?",
    "How long does delivery take?",
    "Is there a warranty on your products?",
    "How do I contact customer support?",
    "I have a problem with my order",
    "Do you offer discounts for bulk orders?",
    "What's your refund policy?",
    "How do I create an account?",
    "Is there a mobile app available?",
    "Where can I find my invoice?",
    "Do you have a loyalty program?",
    "How can I change my password?",
]

class StressTest:
    def __init__(self, base_url, username, password, num_queries=1000, concurrency=10, delay_ms=100):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.token = None
        self.num_queries = num_queries
        self.concurrency = concurrency
        self.delay_ms = delay_ms
        self.results = []
        self.session = None
        self.start_time = None
        self.end_time = None
        self.active_conversation_id = None

    async def login(self):
        """Login and get authentication token"""
        print(f"Logging in as {self.username}...")
        
        async with aiohttp.ClientSession() as session:
            login_data = {
                "username": self.username,
                "password": self.password
            }
            
            try:
                async with session.post(f"{self.base_url}/login", json=login_data) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.token = data.get("access_token")
                        print("Login successful!")
                        return True
                    else:
                        print(f"Login failed with status {response.status}")
                        return False
            except Exception as e:
                print(f"Login error: {str(e)}")
                return False

    async def send_chat_request(self, question, session):
        """Send a chat request and measure response time"""
        if not self.token:
            print("No authentication token available.")
            return None
            
        headers = {"Authorization": f"Bearer {self.token}"}
        start_time = time.time()
        
        try:
            async with session.post(
                f"{self.base_url}/chat",
                json={"question": question},
                headers=headers
            ) as response:
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # Convert to ms
                
                result = {
                    "question": question,
                    "status_code": response.status,
                    "response_time_ms": response_time
                }
                
                if response.status == 200:
                    data = await response.json()
                    self.active_conversation_id = data.get("conversation_id")
                    result["success"] = True
                    result["confidence"] = data.get("confidence")
                    result["was_escalated"] = data.get("escalate", False)
                else:
                    result["success"] = False
                
                return result
        except Exception as e:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            return {
                "question": question,
                "status_code": 0,
                "response_time_ms": response_time,
                "success": False,
                "error": str(e)
            }

    async def worker(self, query_queue, session):
        """Worker that processes queries from the queue"""
        while not query_queue.empty():
            question = await query_queue.get()
            result = await self.send_chat_request(question, session)
            if result:
                self.results.append(result)
            
            # Add a small delay to simulate real user behavior
            await asyncio.sleep(self.delay_ms / 1000)
            
            query_queue.task_done()

    async def run(self):
        """Run the stress test with specified parameters"""
        if not await self.login():
            return False
            
        # Generate questions
        questions = []
        for _ in range(self.num_queries):
            questions.append(random.choice(SAMPLE_QUESTIONS))
        
        # Create a queue with all questions
        query_queue = asyncio.Queue()
        for question in questions:
            query_queue.put_nowait(question)
        
        # Start timing
        self.start_time = time.time()
        print(f"Starting stress test with {self.num_queries} queries and {self.concurrency} concurrent requests...")
        
        # Create session and workers
        self.session = aiohttp.ClientSession()
        workers = []
        for _ in range(min(self.concurrency, self.num_queries)):
            worker = asyncio.create_task(self.worker(query_queue, self.session))
            workers.append(worker)
        
        # Wait for all workers to complete
        await asyncio.gather(*workers)
        await query_queue.join()
        
        # Close session
        await self.session.close()
        
        # End timing
        self.end_time = time.time()
        print("Stress test completed!")
        
        return True

    def generate_report(self):
        """Generate a summary report of test results"""
        if not self.results:
            print("No results to report!")
            return
            
        total_duration = self.end_time - self.start_time
        successful_requests = sum(1 for r in self.results if r.get("success", False))
        failed_requests = len(self.results) - successful_requests
        
        response_times = [r["response_time_ms"] for r in self.results]
        avg_response_time = statistics.mean(response_times) if response_times else 0
        median_response_time = statistics.median(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        
        p95_response_time = 0
        if response_times:
            sorted_times = sorted(response_times)
            p95_index = int(len(sorted_times) * 0.95)
            p95_response_time = sorted_times[p95_index]
        
        # Calculate requests per second
        rps = len(self.results) / total_duration if total_duration > 0 else 0
        
        # Count escalations
        escalated_requests = sum(1 for r in self.results if r.get("was_escalated", False))
        
        # Print report
        print("\n" + "=" * 50)
        print("STRESS TEST REPORT")
        print("=" * 50)
        print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Base URL: {self.base_url}")
        print(f"Total Queries: {self.num_queries}")
        print(f"Concurrent Requests: {self.concurrency}")
        print(f"Request Delay: {self.delay_ms}ms")
        print("-" * 50)
        print(f"Total Duration: {total_duration:.2f} seconds")
        print(f"Successful Requests: {successful_requests} ({successful_requests/len(self.results)*100:.1f}%)")
        print(f"Failed Requests: {failed_requests} ({failed_requests/len(self.results)*100:.1f}%)")
        print(f"Requests Per Second: {rps:.2f}")
        print(f"Escalated Requests: {escalated_requests} ({escalated_requests/successful_requests*100:.1f}% of successful)")
        print("-" * 50)
        print("Response Time Statistics (ms):")
        print(f"  Average: {avg_response_time:.2f}")
        print(f"  Median: {median_response_time:.2f}")
        print(f"  95th Percentile: {p95_response_time:.2f}")
        print(f"  Min: {min_response_time:.2f}")
        print(f"  Max: {max_response_time:.2f}")
        print("=" * 50)
        
        # Save detailed results to a file
        filename = f"stress_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump({
                "test_parameters": {
                    "base_url": self.base_url,
                    "num_queries": self.num_queries,
                    "concurrency": self.concurrency,
                    "delay_ms": self.delay_ms,
                },
                "summary": {
                    "total_duration_seconds": total_duration,
                    "successful_requests": successful_requests,
                    "failed_requests": failed_requests,
                    "requests_per_second": rps,
                    "average_response_time_ms": avg_response_time,
                    "median_response_time_ms": median_response_time,
                    "p95_response_time_ms": p95_response_time,
                    "min_response_time_ms": min_response_time,
                    "max_response_time_ms": max_response_time,
                    "escalated_requests": escalated_requests,
                },
                "detailed_results": self.results
            }, f, indent=2)
        
        print(f"Detailed results saved to {filename}")

async def main():
    parser = argparse.ArgumentParser(description='Run stress test on AI Customer Support System')
    parser.add_argument('--url', type=str, help='Base URL of the API', default="http://localhost:8000")
    parser.add_argument('--username', type=str, help='Username for authentication')
    parser.add_argument('--password', type=str, help='Password for authentication')
    parser.add_argument('--queries', type=int, help='Number of queries to send', default=1000)
    parser.add_argument('--concurrency', type=int, help='Number of concurrent requests', default=10)
    parser.add_argument('--delay', type=int, help='Delay between requests in ms', default=100)
    
    args = parser.parse_args()
    
    # Get credentials from args or environment
    username = args.username or os.getenv("TEST_USERNAME") or "admin"
    password = args.password or os.getenv("TEST_PASSWORD") or "adminpassword"
    
    # Create and run the stress test
    stress_test = StressTest(
        base_url=args.url,
        username=username,
        password=password,
        num_queries=args.queries,
        concurrency=args.concurrency,
        delay_ms=args.delay
    )
    
    if await stress_test.run():
        stress_test.generate_report()
    else:
        print("Test failed to run!")

if __name__ == "__main__":
    asyncio.run(main())
