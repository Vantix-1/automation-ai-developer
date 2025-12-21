"""
Cost Analyzer - Days 6-8
Enhanced cost tracking and analysis for OpenAI API usage
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class CostAnalyzer:
    """Track and analyze OpenAI API costs"""
    
    # Updated pricing (as of Dec 2024)
    PRICING = {
        "gpt-3.5-turbo": {
            "input": 0.0010,  # $0.0010 per 1K tokens
            "output": 0.0020, # $0.0020 per 1K tokens
        },
        "gpt-3.5-turbo-16k": {
            "input": 0.0030,
            "output": 0.0040,
        },
        "gpt-4": {
            "input": 0.0300,
            "output": 0.0600,
        },
        "gpt-4-turbo-preview": {
            "input": 0.0100,
            "output": 0.0300,
        },
        "gpt-4-32k": {
            "input": 0.0600,
            "output": 0.1200,
        },
        "text-embedding-ada-002": {
            "input": 0.00010,  # $0.00010 per 1K tokens
        }
    }
    
    def __init__(self, db_path: str = "cost_tracker.db"):
        """Initialize cost analyzer with SQLite database"""
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create usage table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model TEXT NOT NULL,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                cost REAL DEFAULT 0.0,
                endpoint TEXT,
                user_id TEXT,
                project TEXT
            )
        ''')
        
        # Create daily summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_summary (
                date DATE PRIMARY KEY,
                total_requests INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                total_cost REAL DEFAULT 0.0,
                avg_cost_per_request REAL DEFAULT 0.0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost based on token usage and model"""
        if model not in self.PRICING:
            print(f"⚠️ Pricing not available for {model}, using gpt-3.5-turbo pricing")
            model = "gpt-3.5-turbo"
        
        pricing = self.PRICING[model]
        
        # Calculate input cost
        input_cost = (prompt_tokens / 1000) * pricing.get("input", 0)
        
        # Calculate output cost (if model has output pricing)
        output_cost = 0
        if "output" in pricing:
            output_cost = (completion_tokens / 1000) * pricing["output"]
        
        total_cost = input_cost + output_cost
        
        return round(total_cost, 6)
    
    def log_usage(self, model: str, prompt_tokens: int, completion_tokens: int, 
                  endpoint: str = "chat/completions", user_id: str = "default", 
                  project: str = "default"):
        """Log API usage to database"""
        total_tokens = prompt_tokens + completion_tokens
        cost = self.calculate_cost(model, prompt_tokens, completion_tokens)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO api_usage 
            (model, prompt_tokens, completion_tokens, total_tokens, cost, endpoint, user_id, project)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (model, prompt_tokens, completion_tokens, total_tokens, cost, endpoint, user_id, project))
        
        # Update daily summary
        today = datetime.now().date().isoformat()
        cursor.execute('''
            INSERT OR REPLACE INTO daily_summary (date, total_requests, total_tokens, total_cost, avg_cost_per_request)
            SELECT 
                date(timestamp) as date,
                COUNT(*) as total_requests,
                SUM(total_tokens) as total_tokens,
                SUM(cost) as total_cost,
                AVG(cost) as avg_cost_per_request
            FROM api_usage
            WHERE date(timestamp) = ?
            GROUP BY date(timestamp)
        ''', (today,))
        
        conn.commit()
        conn.close()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
            "endpoint": endpoint
        }
    
    def get_daily_summary(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Get summary for a specific date (default: today)"""
        if not date:
            date = datetime.now().date().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM daily_summary WHERE date = ?
        ''', (date,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "date": result[0],
                "total_requests": result[1],
                "total_tokens": result[2],
                "total_cost": result[3],
                "avg_cost_per_request": result[4]
            }
        else:
            return {
                "date": date,
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "avg_cost_per_request": 0.0
            }
    
    def get_period_summary(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get summary for a date range"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_requests,
                SUM(total_tokens) as total_tokens,
                SUM(cost) as total_cost,
                AVG(cost) as avg_cost_per_request
            FROM api_usage
            WHERE date(timestamp) BETWEEN ? AND ?
        ''', (start_date, end_date))
        
        result = cursor.fetchone()
        
        # Get breakdown by model
        cursor.execute('''
            SELECT 
                model,
                COUNT(*) as requests,
                SUM(prompt_tokens) as prompt_tokens,
                SUM(completion_tokens) as completion_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(cost) as total_cost
            FROM api_usage
            WHERE date(timestamp) BETWEEN ? AND ?
            GROUP BY model
            ORDER BY total_cost DESC
        ''', (start_date, end_date))
        
        model_breakdown = []
        for row in cursor.fetchall():
            model_breakdown.append({
                "model": row[0],
                "requests": row[1],
                "prompt_tokens": row[2],
                "completion_tokens": row[3],
                "total_tokens": row[4],
                "total_cost": row[5]
            })
        
        conn.close()
        
        return {
            "period": f"{start_date} to {end_date}",
            "total_requests": result[0] or 0,
            "total_tokens": result[1] or 0,
            "total_cost": result[2] or 0.0,
            "avg_cost_per_request": result[3] or 0.0,
            "model_breakdown": model_breakdown
        }
    
    def get_usage_trends(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get daily usage trends for specified number of days"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                date,
                total_requests,
                total_tokens,
                total_cost
            FROM daily_summary
            WHERE date BETWEEN ? AND ?
            ORDER BY date
        ''', (start_date.isoformat(), end_date.isoformat()))
        
        trends = []
        for row in cursor.fetchall():
            trends.append({
                "date": row[0],
                "total_requests": row[1],
                "total_tokens": row[2],
                "total_cost": row[3]
            })
        
        conn.close()
        return trends
    
    def generate_report(self, period_days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive usage report"""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=period_days)
        
        period_summary = self.get_period_summary(start_date.isoformat(), end_date.isoformat())
        trends = self.get_usage_trends(period_days)
        
        # Calculate projections
        avg_daily_cost = period_summary["total_cost"] / period_days if period_days > 0 else 0
        monthly_projection = avg_daily_cost * 30
        
        # Cost efficiency metrics
        cost_per_token = period_summary["total_cost"] / period_summary["total_tokens"] if period_summary["total_tokens"] > 0 else 0
        tokens_per_request = period_summary["total_tokens"] / period_summary["total_requests"] if period_summary["total_requests"] > 0 else 0
        
        return {
            "report_period": f"{start_date.isoformat()} to {end_date.isoformat()}",
            "period_days": period_days,
            "summary": period_summary,
            "trends": trends,
            "projections": {
                "avg_daily_cost": round(avg_daily_cost, 4),
                "monthly_projection": round(monthly_projection, 2),
                "estimated_annual_cost": round(monthly_projection * 12, 2)
            },
            "efficiency_metrics": {
                "cost_per_token": round(cost_per_token, 6),
                "tokens_per_request": round(tokens_per_request, 1),
                "cost_per_request": round(period_summary["avg_cost_per_request"], 4)
            },
            "recommendations": self._generate_recommendations(period_summary)
        }
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        if summary["total_cost"] > 10:
            recommendations.append("⚠️  High usage detected. Consider setting daily budget limits.")
        
        # Check model usage
        for model_data in summary.get("model_breakdown", []):
            if "gpt-4" in model_data["model"] and model_data["total_cost"] > 5:
                recommendations.append(f"💡 Consider using gpt-3.5-turbo instead of {model_data['model']} for non-critical tasks (potential savings: ${model_data['total_cost']:.2f})")
        
        if summary["avg_cost_per_request"] > 0.05:
            recommendations.append("💡 Average cost per request is high. Consider optimizing prompts to reduce token usage.")
        
        if len(recommendations) == 0:
            recommendations.append("✅ Usage patterns look efficient. Continue current practices.")
        
        return recommendations
    
    def plot_usage_trends(self, days: int = 30, save_path: Optional[str] = None):
        """Generate visualization of usage trends"""
        trends = self.get_usage_trends(days)
        
        if not trends:
            print("No data to plot")
            return
        
        dates = [t["date"] for t in trends]
        costs = [t["total_cost"] for t in trends]
        tokens = [t["total_tokens"] / 1000 for t in trends]  # Convert to thousands
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Cost plot
        ax1.bar(dates, costs, color='skyblue', alpha=0.7)
        ax1.set_title(f'Daily API Costs (Last {days} Days)')
        ax1.set_ylabel('Cost ($)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add cost labels on bars
        for i, cost in enumerate(costs):
            if cost > 0:
                ax1.text(i, cost, f'${cost:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Token plot
        ax2.plot(dates, tokens, marker='o', linestyle='-', color='orange', linewidth=2)
        ax2.fill_between(dates, tokens, alpha=0.3, color='orange')
        ax2.set_title(f'Daily Token Usage (Last {days} Days)')
        ax2.set_ylabel('Tokens (thousands)')
        ax2.set_xlabel('Date')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Chart saved to {save_path}")
        
        plt.show()
    
    def export_report(self, report: Dict[str, Any], format: str = "json") -> str:
        """Export report in specified format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            filename = f"cost_report_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        
        elif format == "markdown":
            filename = f"cost_report_{timestamp}.md"
            content = self._format_markdown_report(report)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
        
        elif format == "csv":
            filename = f"cost_report_{timestamp}.csv"
            self._export_csv_report(report, filename)
        
        else:
            return f"Unsupported format: {format}"
        
        return filename
    
    def _format_markdown_report(self, report: Dict[str, Any]) -> str:
        """Format report as markdown"""
        content = f"""# OpenAI API Cost Analysis Report

**Report Period:** {report['report_period']}  
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 Executive Summary

- **Total Requests:** {report['summary']['total_requests']:,}
- **Total Tokens:** {report['summary']['total_tokens']:,}
- **Total Cost:** ${report['summary']['total_cost']:.2f}
- **Average Cost/Request:** ${report['summary']['avg_cost_per_request']:.4f}

## 📈 Projections

- **Average Daily Cost:** ${report['projections']['avg_daily_cost']:.2f}
- **Monthly Projection:** ${report['projections']['monthly_projection']:.2f}
- **Annual Projection:** ${report['projections']['estimated_annual_cost']:.2f}

## ⚙️ Efficiency Metrics

- **Cost per Token:** ${report['efficiency_metrics']['cost_per_token']:.6f}
- **Tokens per Request:** {report['efficiency_metrics']['tokens_per_request']:.1f}
- **Cost per Request:** ${report['efficiency_metrics']['cost_per_request']:.4f}

## 🎯 Model Breakdown

"""
        
        for model_data in report['summary']['model_breakdown']:
            content += f"""
### {model_data['model']}
- Requests: {model_data['requests']:,}
- Prompt Tokens: {model_data['prompt_tokens']:,}
- Completion Tokens: {model_data['completion_tokens']:,}
- Total Tokens: {model_data['total_tokens']:,}
- Total Cost: ${model_data['total_cost']:.2f}

"""
        
        content += """## 💡 Recommendations

"""
        
        for recommendation in report['recommendations']:
            content += f"- {recommendation}\n"
        
        return content
    
    def _export_csv_report(self, report: Dict[str, Any], filename: str):
        """Export report as CSV"""
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write summary
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Report Period", report['report_period']])
            writer.writerow(["Total Requests", report['summary']['total_requests']])
            writer.writerow(["Total Tokens", report['summary']['total_tokens']])
            writer.writerow(["Total Cost", f"${report['summary']['total_cost']:.2f}"])
            writer.writerow([])
            
            # Write model breakdown
            writer.writerow(["Model Breakdown"])
            writer.writerow(["Model", "Requests", "Prompt Tokens", "Completion Tokens", "Total Tokens", "Total Cost"])
            for model_data in report['summary']['model_breakdown']:
                writer.writerow([
                    model_data['model'],
                    model_data['requests'],
                    model_data['prompt_tokens'],
                    model_data['completion_tokens'],
                    model_data['total_tokens'],
                    f"${model_data['total_cost']:.2f}"
                ])

def interactive_cost_analyzer():
    """Interactive cost analyzer interface"""
    print("\n" + "="*70)
    print("💰 Advanced Cost Analyzer - Days 6-8")
    print("="*70)
    
    try:
        analyzer = CostAnalyzer()
        
        # Test log some sample data if database is empty
        conn = sqlite3.connect(analyzer.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM api_usage")
        count = cursor.fetchone()[0]
        conn.close()
        
        if count == 0:
            print("\n📝 Database empty. Logging sample data for demonstration...")
            # Log some sample usage
            sample_usage = [
                ("gpt-3.5-turbo", 150, 200),
                ("gpt-3.5-turbo", 200, 150),
                ("gpt-4", 300, 400),
                ("gpt-3.5-turbo", 100, 120),
            ]
            
            for model, prompt, completion in sample_usage:
                analyzer.log_usage(model, prompt, completion)
            print("✅ Sample data logged")
        
        while True:
            print("\n" + "="*50)
            print("📊 Cost Analysis Options:")
            print("1. View Today's Summary")
            print("2. Generate Period Report")
            print("3. View Usage Trends")
            print("4. Plot Usage Charts")
            print("5. Log Manual Usage")
            print("6. Export Report")
            print("7. Exit")
            print("="*50)
            
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == "7":
                print("👋 Goodbye!")
                break
            
            if choice == "1":
                summary = analyzer.get_daily_summary()
                print("\n" + "="*50)
                print(f"📅 Daily Summary - {summary['date']}")
                print("="*50)
                print(f"Total Requests: {summary['total_requests']:,}")
                print(f"Total Tokens: {summary['total_tokens']:,}")
                print(f"Total Cost: ${summary['total_cost']:.4f}")
                print(f"Avg Cost/Request: ${summary['avg_cost_per_request']:.6f}")
            
            elif choice == "2":
                days = int(input("Report period (days, default 7): ") or "7")
                report = analyzer.generate_report(days)
                
                print("\n" + "="*50)
                print(f"📋 {report['period_days']}-Day Cost Report")
                print("="*50)
                
                print(f"\n📊 Summary:")
                print(f"  Period: {report['report_period']}")
                print(f"  Total Requests: {report['summary']['total_requests']:,}")
                print(f"  Total Tokens: {report['summary']['total_tokens']:,}")
                print(f"  Total Cost: ${report['summary']['total_cost']:.2f}")
                
                print(f"\n📈 Projections:")
                print(f"  Avg Daily Cost: ${report['projections']['avg_daily_cost']:.2f}")
                print(f"  Monthly Projection: ${report['projections']['monthly_projection']:.2f}")
                
                print(f"\n🎯 Model Breakdown:")
                for model_data in report['summary']['model_breakdown']:
                    print(f"  {model_data['model']}:")
                    print(f"    Cost: ${model_data['total_cost']:.2f}")
                    print(f"    Tokens: {model_data['total_tokens']:,}")
                    print(f"    Requests: {model_data['requests']}")
                
                print(f"\n💡 Recommendations:")
                for rec in report['recommendations']:
                    print(f"  • {rec}")
            
            elif choice == "3":
                days = int(input("Trend period (days, default 30): ") or "30")
                trends = analyzer.get_usage_trends(days)
                
                if not trends:
                    print("No data available for the specified period")
                    continue
                
                print(f"\n📈 Usage Trends (Last {days} Days)")
                print("="*60)
                print(f"{'Date':<12} {'Requests':<10} {'Tokens':<15} {'Cost':<10}")
                print("-" * 60)
                
                total_cost = 0
                total_tokens = 0
                
                for trend in trends:
                    print(f"{trend['date']:<12} {trend['total_requests']:<10} {trend['total_tokens']:<15,} ${trend['total_cost']:<10.4f}")
                    total_cost += trend['total_cost']
                    total_tokens += trend['total_tokens']
                
                print("-" * 60)
                print(f"{'TOTAL':<12} {'':<10} {total_tokens:<15,} ${total_cost:<10.4f}")
            
            elif choice == "4":
                days = int(input("Chart period (days, default 30): ") or "30")
                save = input("Save chart? (y/n): ").lower()
                save_path = f"usage_chart_{datetime.now().strftime('%Y%m%d')}.png" if save == 'y' else None
                
                analyzer.plot_usage_trends(days, save_path)
            
            elif choice == "5":
                print("\n📝 Log Manual Usage")
                model = input("Model (e.g., gpt-3.5-turbo): ").strip()
                prompt_tokens = int(input("Prompt tokens: ") or "0")
                completion_tokens = int(input("Completion tokens: ") or "0")
                
                if model and (prompt_tokens > 0 or completion_tokens > 0):
                    log = analyzer.log_usage(model, prompt_tokens, completion_tokens)
                    print(f"✅ Usage logged:")
                    print(f"   Model: {log['model']}")
                    print(f"   Tokens: {log['total_tokens']:,} (prompt: {log['prompt_tokens']:,}, completion: {log['completion_tokens']:,})")
                    print(f"   Cost: ${log['cost']:.6f}")
                else:
                    print("❌ Invalid input")
            
            elif choice == "6":
                days = int(input("Report period (days, default 7): ") or "7")
                format_choice = input("Export format (json/markdown/csv, default json): ").strip().lower() or "json"
                
                report = analyzer.generate_report(days)
                filename = analyzer.export_report(report, format_choice)
                print(f"✅ Report exported to {filename}")
            
            else:
                print("❌ Invalid choice")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    interactive_cost_analyzer()