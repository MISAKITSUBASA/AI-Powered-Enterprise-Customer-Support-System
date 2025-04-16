import os
import time
import argparse
import statistics
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabasePerformanceTest:
    def __init__(self, db_url=None, iterations=100):
        # Get database URL from environment or use provided URL
        self.db_url = db_url or os.getenv("DATABASE_URL", "sqlite:///./customer_support.db")
        self.iterations = iterations
        self.results = {
            "select_all_users": [],
            "select_conversations": [],
            "select_messages": [],
            "complex_analytics": [],
            "search_by_content": []
        }
        self.engine = None
        
        # Handle special case for PostgreSQL from Heroku/AWS RDS
        if self.db_url.startswith("postgres://"):
            self.db_url = self.db_url.replace("postgres://", "postgresql://", 1)
            
        print(f"Using database: {self.db_url}")

    def connect(self):
        """Connect to the database"""
        try:
            # Create connection args only for SQLite
            connect_args = {"check_same_thread": False} if self.db_url.startswith("sqlite") else {}
            self.engine = create_engine(self.db_url, connect_args=connect_args)
            
            # Test connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                row = result.fetchone()
                
                if row and row[0] == 1:
                    print("Successfully connected to the database!")
                    return True
                    
        except Exception as e:
            print(f"Error connecting to database: {str(e)}")
            return False
            
        return False

    def run_query(self, query_name, query, params=None):
        """Run a query and measure execution time"""
        if not self.engine:
            print("No database connection.")
            return None
            
        start_time = time.time()
        
        try:
            with self.engine.connect() as conn:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))
                    
                # Fetch all to ensure complete query execution
                rows = result.fetchall()
                
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  # Convert to ms
            
            self.results[query_name].append(execution_time)
            return len(rows)
            
        except Exception as e:
            print(f"Error executing {query_name}: {str(e)}")
            return None

    def run_test(self):
        """Run all database performance tests"""
        if not self.connect():
            return False
            
        print(f"Starting database performance test with {self.iterations} iterations...")
        
        # Test queries
        queries = {
            "select_all_users": "SELECT * FROM users",
            
            "select_conversations": """
                SELECT * FROM conversations 
                WHERE start_time >= :start_date 
                ORDER BY start_time DESC LIMIT 100
            """,
            
            "select_messages": """
                SELECT m.* FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE c.user_id = :user_id
                ORDER BY m.timestamp DESC
                LIMIT 200
            """,
            
            "complex_analytics": """
                SELECT 
                    COUNT(DISTINCT c.id) as conversation_count,
                    COUNT(m.id) as message_count,
                    SUM(CASE WHEN m.was_escalated = 1 THEN 1 ELSE 0 END) as escalated_count,
                    AVG(m.confidence_score) as avg_confidence
                FROM conversations c
                JOIN messages m ON c.id = m.conversation_id
                WHERE c.start_time >= :start_date
                GROUP BY DATE(c.start_time)
                ORDER BY DATE(c.start_time)
            """,
            
            "search_by_content": """
                SELECT m.*, c.user_id
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE m.content LIKE :search_term
                ORDER BY m.timestamp DESC
                LIMIT 50
            """
        }
        
        # Prepare parameters for queries
        thirty_days_ago = datetime.now() - timedelta(days=30)
        params = {
            "select_conversations": {"start_date": thirty_days_ago},
            "select_messages": {"user_id": 1},  # Assuming user ID 1 exists
            "complex_analytics": {"start_date": thirty_days_ago},
            "search_by_content": {"search_term": "%password%"}  # Search for messages containing "password"
        }
        
        # Run test iterations
        for i in range(self.iterations):
            progress = (i + 1) / self.iterations * 100
            print(f"Progress: {progress:.1f}% - Running iteration {i + 1}/{self.iterations}", end="\r")
            
            for query_name, query in queries.items():
                query_params = params.get(query_name)
                self.run_query(query_name, query, query_params)
                
        print("\nTest completed!")
        return True

    def generate_report(self):
        """Generate a summary report of database query performance"""
        if not any(self.results.values()):
            print("No results to report!")
            return
            
        print("\n" + "=" * 60)
        print("DATABASE PERFORMANCE TEST REPORT")
        print("=" * 60)
        print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Database: {self.db_url}")
        print(f"Iterations: {self.iterations}")
        print("-" * 60)
        
        # Print statistics for each query
        for query_name, times in self.results.items():
            if not times:
                continue
                
            avg_time = statistics.mean(times)
            median_time = statistics.median(times)
            min_time = min(times)
            max_time = max(times)
            
            # Calculate 95th percentile
            p95_time = 0
            if times:
                sorted_times = sorted(times)
                p95_index = int(len(sorted_times) * 0.95)
                p95_time = sorted_times[p95_index]
                
            print(f"\nQuery: {query_name}")
            print(f"  Average time: {avg_time:.2f} ms")
            print(f"  Median time: {median_time:.2f} ms")
            print(f"  95th percentile: {p95_time:.2f} ms")
            print(f"  Min time: {min_time:.2f} ms")
            print(f"  Max time: {max_time:.2f} ms")
        
        print("=" * 60)
        
        # Generate graphs
        self.plot_results()
        
    def plot_results(self):
        """Create visualizations of the performance test results"""
        if not any(self.results.values()):
            return
            
        # Set up the figure with a grid of subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Box plot for all queries
        plt.subplot(2, 1, 1)
        data = [times for times in self.results.values() if times]
        labels = [name for name, times in self.results.items() if times]
        plt.boxplot(data, labels=labels, showfliers=True)
        plt.title('Database Query Performance - Box Plot')
        plt.ylabel('Execution Time (ms)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Bar chart with error bars for average times
        plt.subplot(2, 1, 2)
        avg_times = [statistics.mean(times) for name, times in self.results.items() if times]
        std_devs = [statistics.stdev(times) if len(times) > 1 else 0 for name, times in self.results.items() if times]
        x_pos = np.arange(len(labels))
        
        plt.bar(x_pos, avg_times, yerr=std_devs, align='center', alpha=0.7, capsize=10)
        plt.xticks(x_pos, labels, rotation=45, ha='right')
        plt.title('Average Query Execution Times with Standard Deviation')
        plt.ylabel('Execution Time (ms)')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        filename = f"db_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename)
        print(f"Performance graph saved to {filename}")
        
        try:
            plt.show()
        except:
            print("Could not display the plot. The image has been saved to disk.")

def main():
    parser = argparse.ArgumentParser(description='Test database performance')
    parser.add_argument('--db', type=str, help='Database URL (defaults to env var DATABASE_URL)')
    parser.add_argument('--iterations', type=int, help='Number of test iterations', default=100)
    
    args = parser.parse_args()
    
    test = DatabasePerformanceTest(
        db_url=args.db,
        iterations=args.iterations
    )
    
    if test.run_test():
        test.generate_report()
    else:
        print("Test failed to run!")

if __name__ == "__main__":
    main()
