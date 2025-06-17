"""
Render-Ready Deployment Configuration for DGR KPI System
Fixes common deployment issues and provides fallback data
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template_string, request, jsonify
import logging

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Production-ready Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dgr-kpi-dashboard-2024')

# Global variables for system state
system_initialized = False
demo_mode = True  # Enable demo mode for Render deployment
sample_data_loaded = False

# Sample data for demo/fallback mode
SAMPLE_PLANT_DATA = {
    '7MW_Solar': {
        'daily_kpi': pd.DataFrame({
            'date': pd.date_range(start='2024-05-01', end='2024-06-16', freq='D'),
            'energy_export': np.random.normal(25000, 3000, 47).clip(15000, 35000),
            'plant_availability': np.random.normal(94, 3, 47).clip(85, 100),
            'performance_ratio': np.random.normal(87, 4, 47).clip(75, 95),
            'capacity_utilization': np.random.normal(22, 3, 47).clip(15, 30),
            'irradiation_ghi': np.random.normal(5.2, 1.2, 47).clip(2, 8),
            'ambient_temperature': np.random.normal(28, 5, 47).clip(15, 40)
        })
    },
    'NTPC_50MW': {
        'daily_kpi': pd.DataFrame({
            'date': pd.date_range(start='2024-05-01', end='2024-06-16', freq='D'),
            'energy_export': np.random.normal(180000, 25000, 47).clip(120000, 220000),
            'plant_availability': np.random.normal(92, 4, 47).clip(80, 100),
            'performance_ratio': np.random.normal(85, 5, 47).clip(70, 95),
            'capacity_utilization': np.random.normal(25, 4, 47).clip(18, 35),
            'irradiation_ghi': np.random.normal(5.0, 1.1, 47).clip(2, 8),
            'ambient_temperature': np.random.normal(29, 6, 47).clip(15, 42)
        })
    },
    'ESP_Wind_30MW': {
        'daily_kpi': pd.DataFrame({
            'date': pd.date_range(start='2024-05-01', end='2024-06-16', freq='D'),
            'energy_export': np.random.normal(85000, 15000, 47).clip(40000, 120000),
            'plant_availability': np.random.normal(89, 6, 47).clip(75, 100),
            'performance_ratio': np.random.normal(82, 6, 47).clip(65, 95),
            'capacity_utilization': np.random.normal(18, 5, 47).clip(8, 30),
            'wind_speed_avg': np.random.normal(6.2, 2.1, 47).clip(2, 12),
            'ambient_temperature': np.random.normal(27, 5, 47).clip(12, 38)
        })
    },
    'JPPL_70MW': {
        'daily_kpi': pd.DataFrame({
            'date': pd.date_range(start='2024-05-01', end='2024-06-16', freq='D'),
            'energy_export': np.random.normal(250000, 35000, 47).clip(180000, 320000),
            'plant_availability': np.random.normal(95, 3, 47).clip(88, 100),
            'performance_ratio': np.random.normal(88, 3, 47).clip(80, 95),
            'capacity_utilization': np.random.normal(24, 3, 47).clip(18, 32),
            'irradiation_ghi': np.random.normal(5.3, 1.0, 47).clip(3, 8),
            'ambient_temperature': np.random.normal(26, 4, 47).clip(18, 36)
        })
    },
    'AXPPL_40MW': {
        'daily_kpi': pd.DataFrame({
            'date': pd.date_range(start='2024-05-01', end='2024-06-16', freq='D'),
            'energy_export': np.random.normal(150000, 20000, 47).clip(100000, 200000),
            'plant_availability': np.random.normal(93, 4, 47).clip(85, 100),
            'performance_ratio': np.random.normal(86, 4, 47).clip(78, 95),
            'capacity_utilization': np.random.normal(21, 4, 47).clip(15, 30),
            'irradiation_ghi': np.random.normal(5.1, 1.1, 47).clip(2.5, 7.5),
            'ambient_temperature': np.random.normal(30, 6, 47).clip(20, 42)
        })
    }
}

class ProductionDataProcessor:
    """Production-ready data processor with fallback capabilities"""
    
    def __init__(self):
        self.plant_data = {}
        self.data_quality_report = {}
        self.initialized = False
        
    def initialize_with_sample_data(self):
        """Initialize with sample data for demo mode"""
        try:
            logger.info("🎯 Initializing with sample data for demo mode...")
            self.plant_data = SAMPLE_PLANT_DATA.copy()
            
            # Generate quality reports
            for plant_name, data in self.plant_data.items():
                df = data['daily_kpi']
                self.data_quality_report[plant_name] = {
                    'total_rows': len(df),
                    'valid_rows': len(df),
                    'missing_data_summary': {},
                    'column_mapping': dict(zip(df.columns, df.columns)),
                    'date_range': {
                        'start': df['date'].min().strftime('%Y-%m-%d'),
                        'end': df['date'].max().strftime('%Y-%m-%d')
                    }
                }
            
            self.initialized = True
            logger.info(f"✅ Sample data initialized for {len(self.plant_data)} plants")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize sample data: {str(e)}")
            return False
    
    def get_available_plants(self):
        """Get list of available plants"""
        return list(self.plant_data.keys())
    
    def get_plant_data(self, plant_name):
        """Get data for specific plant"""
        if plant_name in self.plant_data:
            return self.plant_data[plant_name]['daily_kpi']
        return None
    
    def get_plant_summary(self, plant_name, days=30):
        """Get plant summary with fallback"""
        try:
            if plant_name not in self.plant_data:
                return None
            
            df = self.plant_data[plant_name]['daily_kpi']
            
            # Filter last N days
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                filtered_df = df[df['date'].dt.date >= start_date]
            else:
                filtered_df = df.tail(days)
            
            if filtered_df.empty:
                return None
            
            summary = {
                'plant_name': plant_name,
                'total_days': len(filtered_df),
                'date_range': f"{start_date} to {end_date}"
            }
            
            # Calculate metrics
            if 'energy_export' in filtered_df.columns:
                energy_data = filtered_df['energy_export'].dropna()
                summary.update({
                    'total_export': float(energy_data.sum()),
                    'avg_daily_export': float(energy_data.mean())
                })
            
            if 'plant_availability' in filtered_df.columns:
                avail_data = filtered_df['plant_availability'].dropna()
                summary['avg_availability'] = float(avail_data.mean())
            
            if 'performance_ratio' in filtered_df.columns:
                pr_data = filtered_df['performance_ratio'].dropna()
                summary['avg_performance_ratio'] = float(pr_data.mean())
            
            if 'capacity_utilization' in filtered_df.columns:
                cuf_data = filtered_df['capacity_utilization'].dropna()
                summary['avg_plf'] = float(cuf_data.mean())
            
            # Data completeness
            total_cells = len(filtered_df) * len(filtered_df.columns)
            non_null_cells = filtered_df.notna().sum().sum()
            summary['data_completeness_pct'] = (non_null_cells / total_cells) * 100
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting plant summary for {plant_name}: {str(e)}")
            return None

class ProductionAIAssistant:
    """Production-ready AI assistant with robust error handling"""
    
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.conversation_memory = []
    
    def process_query(self, user_input):
        """Process user query with comprehensive error handling"""
        try:
            query = user_input.lower().strip()
            
            # Store conversation
            self.conversation_memory.append({
                'timestamp': datetime.now(),
                'query': user_input
            })
            
            # Keep only last 100 conversations
            if len(self.conversation_memory) > 100:
                self.conversation_memory = self.conversation_memory[-100:]
            
            # Route query to appropriate handler
            if 'help' in query:
                return self._handle_help_query()
            elif any(word in query for word in ['status', 'current', 'today', 'now']):
                return self._handle_status_query(query)
            elif any(word in query for word in ['performance', 'efficiency', 'pr']):
                return self._handle_performance_query(query)
            elif any(word in query for word in ['energy', 'generation', 'export', 'kwh']):
                return self._handle_energy_query(query)
            elif any(word in query for word in ['compare', 'vs', 'versus', 'comparison']):
                return self._handle_comparison_query(query)
            elif any(word in query for word in ['financial', 'revenue', 'money', 'cost']):
                return self._handle_financial_query(query)
            elif any(word in query for word in ['alert', 'issue', 'problem', 'warning']):
                return self._handle_alert_query(query)
            elif any(word in query for word in ['summary', 'report', 'overview']):
                return self._handle_summary_query(query)
            else:
                return self._handle_general_query(query)
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return self._format_error_response(str(e))
    
    def _handle_help_query(self):
        """Handle help requests"""
        return """🤖 **DGR KPI Intelligence Assistant Help**

**🚀 What I Can Do:**
• **Status Monitoring**: "What's the status of all plants today?"
• **Performance Analysis**: "Show me performance trends this week"
• **Energy Analysis**: "Total energy generation today"
• **Plant Comparisons**: "Compare 7MW vs NTPC performance"
• **Financial Insights**: "Revenue analysis this month"
• **Alert Management**: "Any critical issues today?"

**💡 Sample Queries:**
• "Portfolio summary"
• "How is NTPC performing?"
• "Energy export from all plants"
• "Compare solar vs wind plants"
• "Financial performance report"
• "Any plants need attention?"

**📊 Available Plants:**
7MW Solar, NTPC 50MW, ESP Wind 30MW, JPPL 70MW, AXPPL 40MW

**🎯 Tips:**
• Be specific with plant names
• Ask follow-up questions for details
• Use natural language - I understand context!

Just ask me anything about your DGR plants! 🚀"""
    
    def _handle_status_query(self, query):
        """Handle status queries"""
        try:
            plants = self.data_processor.get_available_plants()
            
            response = "🔄 **Real-Time Plant Status Dashboard**\n\n"
            response += f"**Portfolio Overview ({len(plants)} plants):**\n"
            
            total_energy = 0
            operational_count = 0
            total_availability = 0
            
            for plant in plants:
                summary = self.data_processor.get_plant_summary(plant, days=1)
                if summary:
                    status_icon = "🟢" if summary.get('avg_availability', 0) > 95 else "🟡" if summary.get('avg_availability', 0) > 85 else "🔴"
                    
                    response += f"\n{status_icon} **{plant}:**\n"
                    response += f"   • Availability: {summary.get('avg_availability', 0):.1f}%\n"
                    response += f"   • Performance: {summary.get('avg_performance_ratio', 0):.1f}%\n"
                    response += f"   • Today's Energy: {summary.get('avg_daily_export', 0):,.0f} kWh\n"
                    
                    total_energy += summary.get('avg_daily_export', 0)
                    if summary.get('avg_availability', 0) > 0:
                        operational_count += 1
                        total_availability += summary.get('avg_availability', 0)
            
            avg_availability = total_availability / max(operational_count, 1)
            
            response += f"\n📊 **Portfolio Summary:**\n"
            response += f"• Total Daily Energy: {total_energy:,.0f} kWh\n"
            response += f"• Operational Plants: {operational_count}/{len(plants)}\n"
            response += f"• Portfolio Availability: {avg_availability:.1f}%\n"
            
            if avg_availability > 95:
                response += "\n✅ **Status**: All systems performing excellently!"
            elif avg_availability > 85:
                response += "\n🟡 **Status**: Good performance with minor attention needed"
            else:
                response += "\n🔴 **Status**: Multiple plants need immediate attention"
            
            return response
            
        except Exception as e:
            return f"❌ Error getting status: {str(e)}"
    
    def _handle_performance_query(self, query):
        """Handle performance analysis queries"""
        try:
            plants = self.data_processor.get_available_plants()
            
            response = "📊 **Performance Analysis Report**\n\n"
            
            performance_data = []
            for plant in plants:
                summary = self.data_processor.get_plant_summary(plant, days=7)
                if summary:
                    performance_data.append({
                        'plant': plant,
                        'availability': summary.get('avg_availability', 0),
                        'performance_ratio': summary.get('avg_performance_ratio', 0),
                        'energy': summary.get('total_export', 0)
                    })
            
            # Sort by performance ratio
            performance_data.sort(key=lambda x: x['performance_ratio'], reverse=True)
            
            response += "🏆 **Performance Rankings (Last 7 days):**\n"
            for i, plant_data in enumerate(performance_data, 1):
                grade = "A" if plant_data['performance_ratio'] > 85 else "B" if plant_data['performance_ratio'] > 80 else "C"
                response += f"\n{i}. **{plant_data['plant']}** (Grade {grade}):\n"
                response += f"   • Performance Ratio: {plant_data['performance_ratio']:.1f}%\n"
                response += f"   • Availability: {plant_data['availability']:.1f}%\n"
                response += f"   • Weekly Energy: {plant_data['energy']:,.0f} kWh\n"
            
            # Calculate portfolio averages
            avg_pr = np.mean([p['performance_ratio'] for p in performance_data])
            avg_avail = np.mean([p['availability'] for p in performance_data])
            
            response += f"\n📈 **Portfolio Averages:**\n"
            response += f"• Performance Ratio: {avg_pr:.1f}%\n"
            response += f"• Availability: {avg_avail:.1f}%\n"
            
            if avg_pr > 85:
                response += "\n✅ **Insight**: Excellent portfolio performance!"
            elif avg_pr > 80:
                response += "\n🟡 **Insight**: Good performance with optimization opportunities"
            else:
                response += "\n🔴 **Insight**: Performance below targets - investigation needed"
            
            return response
            
        except Exception as e:
            return f"❌ Error analyzing performance: {str(e)}"
    
    def _handle_energy_query(self, query):
        """Handle energy analysis queries"""
        try:
            plants = self.data_processor.get_available_plants()
            
            response = "⚡ **Energy Generation Analysis**\n\n"
            
            if 'today' in query or 'daily' in query:
                days = 1
                period = "Today"
            elif 'week' in query:
                days = 7
                period = "This Week"
            elif 'month' in query:
                days = 30
                period = "This Month"
            else:
                days = 7
                period = "Last 7 Days"
            
            energy_data = []
            total_energy = 0
            
            for plant in plants:
                summary = self.data_processor.get_plant_summary(plant, days=days)
                if summary:
                    if days == 1:
                        plant_energy = summary.get('avg_daily_export', 0)
                    else:
                        plant_energy = summary.get('total_export', 0)
                    
                    energy_data.append({
                        'plant': plant,
                        'energy': plant_energy
                    })
                    total_energy += plant_energy
            
            # Sort by energy
            energy_data.sort(key=lambda x: x['energy'], reverse=True)
            
            response += f"🏭 **Energy Production ({period}):**\n"
            for i, plant_data in enumerate(energy_data, 1):
                percentage = (plant_data['energy'] / total_energy * 100) if total_energy > 0 else 0
                response += f"\n{i}. **{plant_data['plant']}**:\n"
                response += f"   • Energy: {plant_data['energy']:,.0f} kWh\n"
                response += f"   • Portfolio Share: {percentage:.1f}%\n"
            
            response += f"\n📊 **Portfolio Totals:**\n"
            response += f"• Total Energy ({period}): {total_energy:,.0f} kWh\n"
            response += f"• Average per Plant: {total_energy/len(energy_data):,.0f} kWh\n"
            
            # Estimated revenue (₹3.50/kWh average)
            estimated_revenue = total_energy * 3.5
            response += f"• Estimated Revenue: ₹{estimated_revenue:,.0f}\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error analyzing energy: {str(e)}"
    
    def _handle_comparison_query(self, query):
        """Handle comparison queries"""
        try:
            plants = self.data_processor.get_available_plants()
            
            response = "⚖️ **Plant Performance Comparison**\n\n"
            
            comparison_data = []
            for plant in plants:
                summary = self.data_processor.get_plant_summary(plant, days=7)
                if summary:
                    comparison_data.append({
                        'plant': plant,
                        'energy': summary.get('total_export', 0),
                        'availability': summary.get('avg_availability', 0),
                        'performance': summary.get('avg_performance_ratio', 0)
                    })
            
            if len(comparison_data) >= 2:
                response += "📊 **Comparative Analysis (Last 7 days):**\n\n"
                response += "| Plant | Energy (kWh) | Availability | Performance |\n"
                response += "|-------|--------------|--------------|-------------|\n"
                
                for plant_data in comparison_data:
                    response += f"| **{plant_data['plant']}** | {plant_data['energy']:,.0f} | {plant_data['availability']:.1f}% | {plant_data['performance']:.1f}% |\n"
                
                # Find best performers
                best_energy = max(comparison_data, key=lambda x: x['energy'])
                best_availability = max(comparison_data, key=lambda x: x['availability'])
                best_performance = max(comparison_data, key=lambda x: x['performance'])
                
                response += f"\n🏆 **Best Performers:**\n"
                response += f"• **Highest Energy**: {best_energy['plant']} ({best_energy['energy']:,.0f} kWh)\n"
                response += f"• **Best Availability**: {best_availability['plant']} ({best_availability['availability']:.1f}%)\n"
                response += f"• **Best Performance**: {best_performance['plant']} ({best_performance['performance']:.1f}%)\n"
                
                # Technology comparison if applicable
                solar_plants = [p for p in comparison_data if 'wind' not in p['plant'].lower()]
                wind_plants = [p for p in comparison_data if 'wind' in p['plant'].lower()]
                
                if solar_plants and wind_plants:
                    solar_avg_pr = np.mean([p['performance'] for p in solar_plants])
                    wind_avg_pr = np.mean([p['performance'] for p in wind_plants])
                    
                    response += f"\n🔬 **Technology Comparison:**\n"
                    response += f"• Solar Average PR: {solar_avg_pr:.1f}%\n"
                    response += f"• Wind Average PR: {wind_avg_pr:.1f}%\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error performing comparison: {str(e)}"
    
    def _handle_financial_query(self, query):
        """Handle financial analysis queries"""
        try:
            plants = self.data_processor.get_available_plants()
            
            response = "💰 **Financial Performance Analysis**\n\n"
            
            # Default tariff rates (₹/kWh)
            tariff_rates = {
                'solar': 3.50,
                'wind': 3.20
            }
            
            financial_data = []
            total_revenue = 0
            total_energy = 0
            
            for plant in plants:
                summary = self.data_processor.get_plant_summary(plant, days=30)
                if summary:
                    energy = summary.get('total_export', 0)
                    tariff = tariff_rates['wind'] if 'wind' in plant.lower() else tariff_rates['solar']
                    revenue = energy * tariff
                    
                    financial_data.append({
                        'plant': plant,
                        'energy': energy,
                        'revenue': revenue,
                        'tariff': tariff,
                        'technology': 'Wind' if 'wind' in plant.lower() else 'Solar'
                    })
                    
                    total_revenue += revenue
                    total_energy += energy
            
            # Sort by revenue
            financial_data.sort(key=lambda x: x['revenue'], reverse=True)
            
            response += f"🏭 **Plant-wise Financial Performance (Last 30 days):**\n"
            for i, plant_data in enumerate(financial_data, 1):
                response += f"\n{i}. **{plant_data['plant']}** ({plant_data['technology']}):\n"
                response += f"   • Energy: {plant_data['energy']:,.0f} kWh\n"
                response += f"   • Revenue: ₹{plant_data['revenue']:,.0f}\n"
                response += f"   • Tariff: ₹{plant_data['tariff']:.2f}/kWh\n"
            
            avg_tariff = total_revenue / total_energy if total_energy > 0 else 0
            
            response += f"\n📊 **Portfolio Financial Summary:**\n"
            response += f"• Total Energy: {total_energy:,.0f} kWh\n"
            response += f"• Total Revenue: ₹{total_revenue:,.0f}\n"
            response += f"• Average Tariff: ₹{avg_tariff:.2f}/kWh\n"
            response += f"• Daily Average Revenue: ₹{total_revenue/30:,.0f}\n"
            response += f"• Annual Projection: ₹{total_revenue*12:,.0f}\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error analyzing financials: {str(e)}"
    
    def _handle_alert_query(self, query):
        """Handle alert and issue queries"""
        try:
            plants = self.data_processor.get_available_plants()
            
            response = "🚨 **Alert & Issue Analysis**\n\n"
            
            critical_alerts = []
            warning_alerts = []
            good_plants = []
            
            for plant in plants:
                summary = self.data_processor.get_plant_summary(plant, days=7)
                if summary:
                    availability = summary.get('avg_availability', 0)
                    performance = summary.get('avg_performance_ratio', 0)
                    
                    if availability < 85 or performance < 75:
                        critical_alerts.append({
                            'plant': plant,
                            'availability': availability,
                            'performance': performance,
                            'issues': []
                        })
                        
                        if availability < 85:
                            critical_alerts[-1]['issues'].append(f"Low availability ({availability:.1f}%)")
                        if performance < 75:
                            critical_alerts[-1]['issues'].append(f"Poor performance ({performance:.1f}%)")
                    
                    elif availability < 95 or performance < 85:
                        warning_alerts.append({
                            'plant': plant,
                            'availability': availability,
                            'performance': performance
                        })
                    else:
                        good_plants.append(plant)
            
            # Critical alerts
            if critical_alerts:
                response += f"🚨 **CRITICAL ALERTS ({len(critical_alerts)} plants):**\n"
                for alert in critical_alerts:
                    response += f"\n• **{alert['plant']}**:\n"
                    for issue in alert['issues']:
                        response += f"  - {issue}\n"
                    response += f"  - **Action**: Immediate investigation required\n"
            
            # Warning alerts
            if warning_alerts:
                response += f"\n⚠️ **WARNING ALERTS ({len(warning_alerts)} plants):**\n"
                for alert in warning_alerts:
                    response += f"• **{alert['plant']}**: Monitor closely (Avail: {alert['availability']:.1f}%, PR: {alert['performance']:.1f}%)\n"
            
            # Good plants
            if good_plants:
                response += f"\n✅ **PERFORMING WELL ({len(good_plants)} plants):**\n"
                response += f"{', '.join(good_plants)}\n"
            
            # Summary
            total_plants = len(plants)
            operational_rate = ((total_plants - len(critical_alerts)) / total_plants) * 100
            
            response += f"\n📊 **Alert Summary:**\n"
            response += f"• Critical Issues: {len(critical_alerts)} plants\n"
            response += f"• Warnings: {len(warning_alerts)} plants\n"
            response += f"• Operational Rate: {operational_rate:.1f}%\n"
            
            if operational_rate > 90:
                response += "\n✅ **Overall Status**: Portfolio operating well"
            elif operational_rate > 75:
                response += "\n🟡 **Overall Status**: Some attention needed"
            else:
                response += "\n🔴 **Overall Status**: Multiple issues require immediate action"
            
            return response
            
        except Exception as e:
            return f"❌ Error analyzing alerts: {str(e)}"
    
    def _handle_summary_query(self, query):
        """Handle summary and report queries"""
        try:
            plants = self.data_processor.get_available_plants()
            
            response = "📋 **Executive Portfolio Summary**\n\n"
            
            # Portfolio metrics
            total_energy = 0
            total_availability = 0
            total_performance = 0
            operational_plants = 0
            
            plant_summaries = []
            
            for plant in plants:
                summary = self.data_processor.get_plant_summary(plant, days=30)
                if summary:
                    plant_summaries.append({
                        'plant': plant,
                        'energy': summary.get('total_export', 0),
                        'availability': summary.get('avg_availability', 0),
                        'performance': summary.get('avg_performance_ratio', 0)
                    })
                    
                    total_energy += summary.get('total_export', 0)
                    if summary.get('avg_availability', 0) > 0:
                        operational_plants += 1
                        total_availability += summary.get('avg_availability', 0)
                        total_performance += summary.get('avg_performance_ratio', 0)
            
            avg_availability = total_availability / max(operational_plants, 1)
            avg_performance = total_performance / max(operational_plants, 1)
            
            response += f"📊 **Portfolio Metrics (Last 30 days):**\n"
            response += f"• Total Energy Export: {total_energy:,.0f} kWh\n"
            response += f"• Portfolio Availability: {avg_availability:.1f}%\n"
            response += f"• Portfolio Performance: {avg_performance:.1f}%\n"
            response += f"• Operational Plants: {operational_plants}/{len(plants)}\n"
            
            # Estimated revenue
            estimated_revenue = total_energy * 3.4
            response += f"• Estimated Revenue: ₹{estimated_revenue:,.0f}\n"
            
            # Top performers
            plant_summaries.sort(key=lambda x: x['energy'], reverse=True)
            response += f"\n🏆 **Top 3 Energy Producers:**\n"
            for i, plant in enumerate(plant_summaries[:3], 1):
                response += f"{i}. {plant['plant']}: {plant['energy']:,.0f} kWh\n"
            
            # Performance grades
            excellent = [p for p in plant_summaries if p['performance'] > 85]
            good = [p for p in plant_summaries if 80 <= p['performance'] <= 85]
            needs_attention = [p for p in plant_summaries if p['performance'] < 80]
            
            response += f"\n📊 **Performance Distribution:**\n"
            response += f"• Excellent (>85% PR): {len(excellent)} plants\n"
            response += f"• Good (80-85% PR): {len(good)} plants\n"
            response += f"• Needs Attention (<80% PR): {len(needs_attention)} plants\n"
            
            # Key insights
            response += f"\n💡 **Key Insights:**\n"
            if avg_performance > 85:
                response += "• Portfolio performing excellently\n"
            elif avg_performance > 80:
                response += "• Good portfolio performance with optimization opportunities\n"
            else:
                response += "• Portfolio performance below targets - action needed\n"
                
            if avg_availability > 95:
                response += "• Excellent availability across portfolio\n"
            elif avg_availability > 90:
                response += "• Good availability with room for improvement\n"
            else:
                response += "• Availability issues need immediate attention\n"
            
            # Next steps
            response += f"\n🎯 **Recommended Next Steps:**\n"
            if needs_attention:
                response += f"• Focus on improving {len(needs_attention)} underperforming plants\n"
            response += "• Continue monitoring daily performance trends\n"
            response += "• Review monthly financial performance vs targets\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error generating summary: {str(e)}"
    
    def _handle_general_query(self, query):
        """Handle general queries with intelligent suggestions"""
        response = f"🤔 **I understand you're asking about:** *\"{query}\"*\n\n"
        
        response += "💡 **Here are some things I can help you with:**\n\n"
        
        response += "**📊 Status & Monitoring:**\n"
        response += "• 'What's the status of all plants today?'\n"
        response += "• 'Any plants down right now?'\n"
        response += "• 'Current operational status'\n\n"
        
        response += "**📈 Performance Analysis:**\n"
        response += "• 'Performance analysis report'\n"
        response += "• 'How is NTPC performing this week?'\n"
        response += "• 'Show me efficiency trends'\n\n"
        
        response += "**⚡ Energy & Generation:**\n"
        response += "• 'Total energy generation today'\n"
        response += "• 'Energy export from all plants'\n"
        response += "• 'Monthly energy production'\n\n"
        
        response += "**💰 Financial Analysis:**\n"
        response += "• 'Revenue analysis this month'\n"
        response += "• 'Financial performance summary'\n"
        response += "• 'ROI calculation by plant'\n\n"
        
        response += "**🔍 Comparisons:**\n"
        response += "• 'Compare 7MW vs NTPC performance'\n"
        response += "• 'Best performing plant this month'\n"
        response += "• 'Solar vs wind comparison'\n\n"
        
        plants = self.data_processor.get_available_plants()
        response += f"**🏭 Available Plants:** {', '.join(plants[:3])}{'...' if len(plants) > 3 else ''}\n\n"
        
        response += "Just ask me anything about your DGR plants in natural language! 🚀"
        
        return response
    
    def _format_error_response(self, error):
        """Format user-friendly error response"""
        return f"""🤔 **I encountered an issue processing your request**

**Error Details**: {error}

**🔧 Let's try these solutions:**
• **Rephrase your question** - Try asking in different words
• **Be more specific** - Mention exact plant names or time periods
• **Use simple queries** - Break complex questions into parts

**💡 Quick examples that work well:**
• "Show me plant status"
• "Performance summary"
• "Energy generation report"
• "Compare all plants"

**🆘 Still need help?** Try asking 'help' to see my full capabilities!"""

# Initialize global components
data_processor = None
ai_assistant = None

def initialize_system():
    """Initialize the complete system with error handling"""
    global data_processor, ai_assistant, system_initialized, sample_data_loaded
    
    try:
        logger.info("🚀 Initializing DGR KPI System for Render deployment...")
        
        # Initialize data processor
        data_processor = ProductionDataProcessor()
        
        # Try to load real data first, fallback to sample data
        try:
            # This would be your real data loading logic
            # real_data_loaded = load_real_excel_data()
            real_data_loaded = False  # Set to True when you have real data
            
            if not real_data_loaded:
                logger.info("📊 Loading sample data for demo mode...")
                success = data_processor.initialize_with_sample_data()
                if success:
                    sample_data_loaded = True
                    logger.info("✅ Sample data loaded successfully")
                else:
                    raise Exception("Failed to load sample data")
            
        except Exception as e:
            logger.error(f"❌ Data loading failed: {str(e)}")
            return False
        
        # Initialize AI assistant
        ai_assistant = ProductionAIAssistant(data_processor)
        
        system_initialized = True
        logger.info("✅ DGR KPI System initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {str(e)}")
        system_initialized = False
        return False

# Flask routes
@app.route('/')
def dashboard():
    """Serve the main dashboard"""
    # Initialize system on first request
    if not system_initialized:
        initialize_system()
    
    return render_template_string(DASHBOARD_HTML_TEMPLATE)

@app.route('/api/dashboard')
def api_dashboard():
    """Dashboard API endpoint"""
    try:
        if not system_initialized:
            initialize_system()
        
        if not data_processor:
            return jsonify({'error': 'System not initialized'}), 500
        
        plants = data_processor.get_available_plants()
        
        # Calculate portfolio metrics
        portfolio_metrics = {
            'total_plants': len(plants),
            'operational_plants': 0,
            'total_energy': 0,
            'avg_availability': 0,
            'avg_performance': 0
        }
        
        plant_summaries = []
        availability_sum = 0
        performance_sum = 0
        operational_count = 0
        
        for plant in plants:
            summary = data_processor.get_plant_summary(plant, days=1)
            if summary:
                availability = summary.get('avg_availability', 0)
                performance = summary.get('avg_performance_ratio', 0)
                energy = summary.get('avg_daily_export', 0)
                
                status = 'excellent' if availability > 95 else 'good' if availability > 85 else 'warning' if availability > 70 else 'critical'
                
                plant_summaries.append({
                    'name': plant,
                    'status': status,
                    'energy': energy,
                    'availability': availability,
                    'performance': performance,
                    'data_quality': 98.5,  # Demo value
                    'trend': 'stable',
                    'last_update': datetime.now().strftime('%Y-%m-%d')
                })
                
                portfolio_metrics['total_energy'] += energy
                if availability > 0:
                    operational_count += 1
                    availability_sum += availability
                    performance_sum += performance
        
        portfolio_metrics['operational_plants'] = operational_count
        if operational_count > 0:
            portfolio_metrics['avg_availability'] = availability_sum / operational_count
            portfolio_metrics['avg_performance'] = performance_sum / operational_count
        
        # Generate alerts
        system_alerts = []
        for plant_data in plant_summaries:
            if plant_data['status'] == 'critical':
                system_alerts.append(f"🚨 {plant_data['name']}: Critical availability ({plant_data['availability']:.1f}%)")
            elif plant_data['status'] == 'warning':
                system_alerts.append(f"⚠️ {plant_data['name']}: Below target availability ({plant_data['availability']:.1f}%)")
        
        if not system_alerts:
            system_alerts.append("✅ All plants operating within normal parameters")
        
        return jsonify({
            'portfolio_metrics': portfolio_metrics,
            'plant_summaries': plant_summaries,
            'system_alerts': system_alerts,
            'system_status': {
                'initialized': True,
                'plants_loaded': len(plants),
                'ai_ready': True,
                'demo_mode': demo_mode
            },
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"Dashboard API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """Chat API endpoint"""
    try:
        if not system_initialized:
            initialize_system()
        
        if not ai_assistant:
            return jsonify({
                'response': '🤖 AI Assistant is initializing. Please wait a moment and try again.',
                'error': True
            })
        
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'response': 'Please provide a message to process.',
                'error': True
            })
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({
                'response': 'Please ask me something about your DGR plants!',
                'error': True
            })
        
        logger.info(f"Processing chat query: {user_message[:50]}...")
        
        # Process with AI assistant
        response = ai_assistant.process_query(user_message)
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error': False,
            'demo_mode': demo_mode
        })
        
    except Exception as e:
        logger.error(f"Chat API error: {str(e)}")
        return jsonify({
            'response': f'I encountered a technical issue: {str(e)}. Please try again with a simpler question.',
            'error': True
        })

@app.route('/api/health')
def api_health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if system_initialized else 'initializing',
        'demo_mode': demo_mode,
        'sample_data_loaded': sample_data_loaded,
        'timestamp': datetime.now().isoformat()
    })

# Dashboard HTML Template
DASHBOARD_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DGR KPI Intelligence Dashboard</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            --warning-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
            --danger-gradient: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
            --glass-bg: rgba(255, 255, 255, 0.1);
            --glass-border: rgba(255, 255, 255, 0.2);
            --shadow-soft: 0 8px 32px rgba(31, 38, 135, 0.37);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--primary-gradient);
            min-height: 100vh;
            color: #2c3e50;
        }

        .dashboard {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            background: var(--glass-bg);
            backdrop-filter: blur(15px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: var(--shadow-soft);
        }

        .header h1 {
            font-size: 2.5em;
            font-weight: 700;
            color: white;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: var(--success-gradient);
            color: white;
            padding: 8px 16px;
            border-radius: 25px;
            font-size: 0.9em;
            font-weight: 600;
            margin-left: 15px;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }

        .metric-card {
            background: var(--glass-bg);
            backdrop-filter: blur(15px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 30px;
            box-shadow: var(--shadow-soft);
            transition: all 0.3s ease;
        }

        .metric-card:hover {
            transform: translateY(-5px);
        }

        .metric-icon {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: var(--success-gradient);
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
            color: white;
            font-size: 1.5em;
        }

        .metric-title {
            font-size: 1em;
            color: rgba(255, 255, 255, 0.8);
            margin-bottom: 15px;
            font-weight: 500;
        }

        .metric-value {
            font-size: 2.5em;
            font-weight: 700;
            color: white;
            margin-bottom: 10px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        }

        .metric-unit {
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.9em;
            font-weight: 500;
        }

        .main-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }

        .content-section {
            background: var(--glass-bg);
            backdrop-filter: blur(15px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 30px;
            box-shadow: var(--shadow-soft);
        }

        .section-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid rgba(255,255,255,0.1);
        }

        .section-icon {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: var(--success-gradient);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
        }

        .section-title {
            font-size: 1.4em;
            font-weight: 600;
            color: white;
        }

        .plant-grid {
            display: grid;
            gap: 15px;
        }

        .plant-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
        }

        .plant-card:hover {
            background: rgba(255, 255, 255, 0.1);
            transform: translateX(5px);
        }

        .plant-status-indicator {
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 4px;
            border-radius: 15px 0 0 15px;
        }

        .plant-card.excellent .plant-status-indicator { background: #4CAF50; }
        .plant-card.good .plant-status-indicator { background: #8BC34A; }
        .plant-card.warning .plant-status-indicator { background: #FF9800; }
        .plant-card.critical .plant-status-indicator { background: #F44336; }

        .plant-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .plant-name {
            font-size: 1.1em;
            font-weight: 600;
            color: white;
        }

        .plant-status-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 500;
        }

        .plant-status-badge.excellent { background: rgba(76, 175, 80, 0.2); color: #4CAF50; }
        .plant-status-badge.good { background: rgba(139, 195, 74, 0.2); color: #8BC34A; }
        .plant-status-badge.warning { background: rgba(255, 152, 0, 0.2); color: #FF9800; }
        .plant-status-badge.critical { background: rgba(244, 67, 54, 0.2); color: #F44336; }

        .plant-metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            font-size: 0.9em;
        }

        .plant-metric {
            text-align: center;
        }

        .plant-metric-value {
            font-size: 1.2em;
            font-weight: 600;
            color: white;
            margin-bottom: 5px;
        }

        .plant-metric-label {
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.8em;
        }

        .alerts-container {
            max-height: 400px;
            overflow-y: auto;
        }

        .alert-item {
            background: rgba(255, 255, 255, 0.05);
            border-left: 4px solid;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            transition: all 0.3s ease;
        }

        .alert-item:hover {
            background: rgba(255, 255, 255, 0.1);
        }

        .alert-item.critical { border-left-color: #F44336; }
        .alert-item.warning { border-left-color: #FF9800; }
        .alert-item.info { border-left-color: #2196F3; }

        .alert-content {
            color: white;
            line-height: 1.4;
        }

        .chat-section {
            grid-column: 1 / -1;
            background: var(--glass-bg);
            backdrop-filter: blur(15px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 30px;
            box-shadow: var(--shadow-soft);
            height: 600px;
            display: flex;
            flex-direction: column;
        }

        .chat-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid rgba(255,255,255,0.1);
        }

        .ai-avatar {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: var(--success-gradient);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.5em;
        }

        .chat-title {
            flex: 1;
        }

        .chat-title h3 {
            color: white;
            font-size: 1.3em;
            margin-bottom: 5px;
        }

        .chat-subtitle {
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.9em;
        }

        .quick-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 20px;
        }

        .quick-action {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
            padding: 8px 16px;
            border-radius: 25px;
            font-size: 0.9em;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .quick-action:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }

        .chat-messages {
            flex: 1;
            background: rgba(0, 0, 0, 0.1);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            overflow-y: auto;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .message {
            margin-bottom: 20px;
            padding: 15px 20px;
            border-radius: 20px;
            max-width: 85%;
            line-height: 1.5;
            animation: messageSlide 0.3s ease-out;
        }

        @keyframes messageSlide {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            background: var(--success-gradient);
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 5px;
        }

        .message.ai {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-bottom-left-radius: 5px;
            white-space: pre-wrap;
        }

        .message-time {
            font-size: 0.7em;
            opacity: 0.7;
            margin-top: 5px;
        }

        .chat-input {
            display: flex;
            gap: 15px;
            align-items: center;
        }

        .chat-input-field {
            flex: 1;
            background: rgba(255, 255, 255, 0.1);
            border: 2px solid rgba(255, 255, 255, 0.2);
            border-radius: 25px;
            padding: 15px 20px;
            color: white;
            font-size: 1em;
            outline: none;
            transition: all 0.3s ease;
        }

        .chat-input-field:focus {
            border-color: rgba(255, 255, 255, 0.5);
            background: rgba(255, 255, 255, 0.15);
        }

        .chat-input-field::placeholder {
            color: rgba(255, 255, 255, 0.6);
        }

        .send-button {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background: var(--success-gradient);
            border: none;
            color: white;
            font-size: 1.2em;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .send-button:hover {
            transform: scale(1.1);
        }

        .send-button:disabled {
            background: rgba(255, 255, 255, 0.2);
            cursor: not-allowed;
            transform: none;
        }

        .demo-banner {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            color: white;
            text-align: center;
            padding: 10px;
            font-weight: 600;
            margin-bottom: 20px;
            border-radius: 10px;
        }

        @media (max-width: 1024px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .metrics-grid {
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            }
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <!-- Demo Banner -->
        <div class="demo-banner" id="demoBanner" style="display: none;">
            🚀 DEMO MODE: Using sample data for demonstration. Deploy with your Excel files for live data.
        </div>

        <!-- Header Section -->
        <div class="header">
            <h1>
                <i class="fas fa-solar-panel"></i>
                DGR KPI Intelligence
                <span class="status-indicator" id="statusIndicator">
                    <i class="fas fa-circle"></i>
                    <span id="statusText">Initializing...</span>
                </span>
            </h1>
            <div style="color: rgba(255,255,255,0.9); font-size: 1.1em; margin-top: 15px;">
                <span>Intelligent Analytics • Real-time Monitoring • Predictive Insights</span>
                <span id="lastUpdate" style="float: right;">System starting...</span>
            </div>
        </div>

        <!-- Metrics Grid -->
        <div class="metrics-grid" id="metricsGrid">
            <div class="metric-card">
                <div class="metric-icon"><i class="fas fa-bolt"></i></div>
                <div class="metric-title">System Status</div>
                <div class="metric-value">Loading...</div>
                <div class="metric-unit">Initializing components</div>
            </div>
        </div>

        <!-- Main Content -->
        <div class="main-content">
            <!-- Plant Performance Section -->
            <div class="content-section">
                <div class="section-header">
                    <div class="section-icon">
                        <i class="fas fa-industry"></i>
                    </div>
                    <div>
                        <div class="section-title">Plant Performance</div>
                        <div style="color: rgba(255,255,255,0.7); font-size: 0.9em;">Real-time monitoring</div>
                    </div>
                </div>
                <div class="plant-grid" id="plantsContainer">
                    <div style="text-align: center; color: rgba(255,255,255,0.7); padding: 40px;">
                        <i class="fas fa-spinner fa-spin" style="font-size: 2em; margin-bottom: 15px;"></i>
                        <div>Loading plant performance data...</div>
                    </div>
                </div>
            </div>

            <!-- Alerts Section -->
            <div class="content-section">
                <div class="section-header">
                    <div class="section-icon">
                        <i class="fas fa-exclamation-triangle"></i>
                    </div>
                    <div>
                        <div class="section-title">Smart Alerts</div>
                        <div style="color: rgba(255,255,255,0.7); font-size: 0.9em;">AI-powered notifications</div>
                    </div>
                </div>
                <div class="alerts-container" id="alertsContainer">
                    <div style="text-align: center; color: rgba(255,255,255,0.7); padding: 20px;">
                        <i class="fas fa-search" style="font-size: 1.5em; margin-bottom: 10px;"></i>
                        <div>Analyzing system for alerts...</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- AI Chat Section -->
        <div class="chat-section">
            <div class="chat-header">
                <div class="ai-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="chat-title">
                    <h3>AI Assistant</h3>
                    <div class="chat-subtitle">Your intelligent DGR operations companion</div>
                </div>
            </div>

            <!-- Quick Actions -->
            <div class="quick-actions">
                <div class="quick-action" onclick="sendQuickQuery('portfolio summary')">
                    <i class="fas fa-chart-line"></i> Portfolio Summary
                </div>
                <div class="quick-action" onclick="sendQuickQuery('plant status today')">
                    <i class="fas fa-industry"></i> Plant Status
                </div>
                <div class="quick-action" onclick="sendQuickQuery('performance analysis')">
                    <i class="fas fa-tachometer-alt"></i> Performance Analysis
                </div>
                <div class="quick-action" onclick="sendQuickQuery('energy generation report')">
                    <i class="fas fa-bolt"></i> Energy Report
                </div>
                <div class="quick-action" onclick="sendQuickQuery('financial analysis')">
                    <i class="fas fa-money-bill-wave"></i> Financial Analysis
                </div>
                <div class="quick-action" onclick="sendQuickQuery('help')">
                    <i class="fas fa-question-circle"></i> Help
                </div>
            </div>

            <!-- Chat Messages -->
            <div class="chat-messages" id="chatMessages">
                <div class="message ai">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                        <i class="fas fa-robot" style="color: #4CAF50;"></i>
                        <strong>Your AI Assistant is Ready!</strong>
                    </div>
                    
                    Welcome to the DGR KPI Intelligence Dashboard! I'm here to help you monitor and analyze your renewable energy plants.

                    <div style="margin: 15px 0;">
                        <strong>🚀 Quick Start:</strong><br>
                        • Ask me about plant status, performance, or energy generation<br>
                        • Use the quick action buttons above for common queries<br>
                        • I understand natural language - just ask normally!<br>
                    </div>

                    <div style="margin: 15px 0;">
                        <strong>💡 Example Queries:</strong><br>
                        • "What's the status of all plants today?"<br>
                        • "Show me performance analysis"<br>
                        • "Compare energy generation across plants"<br>
                        • "Any critical alerts I should know about?"<br>
                    </div>

                    <strong>Ready to help with your DGR plant operations! 🌟</strong>
                    <div class="message-time">${new Date().toLocaleTimeString()}</div>
                </div>
            </div>

            <!-- Chat Input -->
            <div class="chat-input">
                <input type="text" class="chat-input-field" id="chatInput" 
                       placeholder="Ask me about plant performance, alerts, comparisons, financial analysis...">
                <button class="send-button" id="sendButton" onclick="sendMessage()">
                    <i class="fas fa-paper-plane"></i>
                </button>
            </div>
        </div>
    </div>

    <script>
        // Global variables
        let isProcessing = false;
        
        // Update dashboard data
        function updateDashboard() {
            fetch('/api/dashboard')
                .then(response => response.json())
                .then(data => {
                    updateMetrics(data.portfolio_metrics || {});
                    updatePlants(data.plant_summaries || []);
                    updateAlerts(data.system_alerts || []);
                    updateSystemStatus(data.system_status || {});
                    updateTimestamp(data.last_updated);
                    
                    // Show demo banner if in demo mode
                    if (data.system_status && data.system_status.demo_mode) {
                        document.getElementById('demoBanner').style.display = 'block';
                    }
                })
                .catch(error => {
                    console.error('Dashboard update error:', error);
                    updateSystemStatus({ initialized: false, error_message: 'Connection failed' });
                });
        }

        function updateMetrics(metrics) {
            const grid = document.getElementById('metricsGrid');
            grid.innerHTML = `
                <div class="metric-card">
                    <div class="metric-icon"><i class="fas fa-bolt"></i></div>
                    <div class="metric-title">Total Energy Export</div>
                    <div class="metric-value">${formatNumber(metrics.total_energy || 0)}</div>
                    <div class="metric-unit">kWh (Portfolio Total)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon"><i class="fas fa-percentage"></i></div>
                    <div class="metric-title">Portfolio Availability</div>
                    <div class="metric-value">${(metrics.avg_availability || 0).toFixed(1)}%</div>
                    <div class="metric-unit">Average Uptime</div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon"><i class="fas fa-tachometer-alt"></i></div>
                    <div class="metric-title">Performance Ratio</div>
                    <div class="metric-value">${(metrics.avg_performance || 0).toFixed(1)}%</div>
                    <div class="metric-unit">System Efficiency</div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon"><i class="fas fa-industry"></i></div>
                    <div class="metric-title">Plant Status</div>
                    <div class="metric-value">${metrics.operational_plants || 0}/${metrics.total_plants || 0}</div>
                    <div class="metric-unit">Operational Plants</div>
                </div>
            `;
        }

        function updatePlants(plants) {
            const container = document.getElementById('plantsContainer');
            if (plants.length === 0) {
                container.innerHTML = `
                    <div style="text-align: center; color: rgba(255,255,255,0.7); padding: 40px;">
                        <i class="fas fa-spinner fa-spin" style="font-size: 2em; margin-bottom: 15px;"></i>
                        <div>Loading plant performance data...</div>
                    </div>
                `;
                return;
            }

            container.innerHTML = plants.map(plant => `
                <div class="plant-card ${plant.status}" onclick="queryPlant('${plant.name}')">
                    <div class="plant-status-indicator"></div>
                    <div class="plant-header">
                        <div class="plant-name">
                            <i class="fas fa-solar-panel"></i> ${plant.name}
                        </div>
                        <div class="plant-status-badge ${plant.status}">
                            ${getStatusText(plant.status)}
                        </div>
                    </div>
                    <div class="plant-metrics">
                        <div class="plant-metric">
                            <div class="plant-metric-value">${plant.availability.toFixed(1)}%</div>
                            <div class="plant-metric-label">Availability</div>
                        </div>
                        <div class="plant-metric">
                            <div class="plant-metric-value">${plant.performance.toFixed(1)}%</div>
                            <div class="plant-metric-label">Performance</div>
                        </div>
                        <div class="plant-metric">
                            <div class="plant-metric-value">${formatNumber(plant.energy)}</div>
                            <div class="plant-metric-label">Energy (kWh)</div>
                        </div>
                    </div>
                </div>
            `).join('');
        }

        function updateAlerts(alerts) {
            const container = document.getElementById('alertsContainer');
            if (alerts.length === 0) {
                container.innerHTML = `
                    <div class="alert-item info">
                        <div class="alert-content">
                            <i class="fas fa-check-circle" style="color: #4CAF50; margin-right: 10px;"></i>
                            <strong>All Systems Operational</strong><br>
                            No critical alerts detected. All plants operating within normal parameters.
                        </div>
                    </div>
                `;
                return;
            }

            container.innerHTML = alerts.map(alert => {
                const alertType = alert.includes('🚨') ? 'critical' : 
                                alert.includes('⚠️') ? 'warning' : 'info';
                
                return `
                    <div class="alert-item ${alertType}">
                        <div class="alert-content">${alert}</div>
                    </div>
                `;
            }).join('');
        }

        function updateSystemStatus(status) {
            const indicator = document.getElementById('statusIndicator');
            const statusText = document.getElementById('statusText');
            
            if (status.initialized) {
                indicator.className = 'status-indicator';
                statusText.textContent = `Online (${status.plants_loaded || 0} plants)`;
            } else {
                indicator.className = 'status-indicator error';
                statusText.textContent = status.error_message || 'System Error';
            }
        }

        function updateTimestamp(timestamp) {
            if (timestamp) {
                document.getElementById('lastUpdate').textContent = `Last updated: ${timestamp}`;
            }
        }

        function getStatusText(status) {
            const statusMap = {
                'excellent': '🟢 Excellent',
                'good': '🟡 Good', 
                'warning': '🟠 Warning',
                'critical': '🔴 Critical'
            };
            return statusMap[status] || status;
        }

        function formatNumber(num) {
            if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
            if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
            return num.toLocaleString();
        }

        // Chat functionality
        function sendMessage() {
            const input = document.getElementById('chatInput');
            const button = document.getElementById('sendButton');
            const message = input.value.trim();
            
            if (!message || isProcessing) return;
            
            isProcessing = true;
            
            // Add user message
            addMessage('user', message);
            
            input.value = '';
            button.disabled = true;
            button.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            
            // Send to AI
            fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message })
            })
            .then(response => response.json())
            .then(data => {
                addMessage('ai', data.response);
            })
            .catch(error => {
                addMessage('ai', '❌ I apologize, but I encountered an error processing your request. Please try again.');
                console.error('Chat error:', error);
            })
            .finally(() => {
                isProcessing = false;
                button.disabled = false;
                button.innerHTML = '<i class="fas fa-paper-plane"></i>';
                input.focus();
            });
        }

        function sendQuickQuery(query) {
            document.getElementById('chatInput').value = query;
            sendMessage();
        }

        function queryPlant(plantName) {
            const query = `Tell me about ${plantName} performance and status`;
            document.getElementById('chatInput').value = query;
            sendMessage();
        }

        function addMessage(type, content) {
            const chatMessages = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            
            if (type === 'ai') {
                messageDiv.innerHTML = `
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                        <i class="fas fa-robot" style="color: #4CAF50;"></i>
                        <strong>AI Assistant</strong>
                    </div>
                    ${content}
                    <div class="message-time">${new Date().toLocaleTimeString()}</div>
                `;
            } else {
                messageDiv.innerHTML = `
                    ${content}
                    <div class="message-time">${new Date().toLocaleTimeString()}</div>
                `;
            }
            
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        // Keyboard event handlers
        document.getElementById('chatInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !isProcessing) {
                sendMessage();
            }
        });

        // Initialize dashboard
        function initializeDashboard() {
            console.log('🚀 Initializing DGR Dashboard');
            
            // Initial dashboard update
            updateDashboard();
            
            // Set up periodic updates
            setInterval(updateDashboard, 60000); // Update every minute
            
            // Focus on chat input
            setTimeout(() => {
                document.getElementById('chatInput').focus();
            }, 1000);
            
            console.log('✅ Dashboard initialized successfully');
        }

        // Start dashboard when page loads
        document.addEventListener('DOMContentLoaded', initializeDashboard);
    </script>
</body>
</html>
"""

# Environment-specific configuration
def get_config():
    """Get configuration based on environment"""
    return {
        'DEBUG': os.environ.get('FLASK_ENV') != 'production',
        'PORT': int(os.environ.get('PORT', 5000)),
        'HOST': '0.0.0.0',
        'SECRET_KEY': os.environ.get('SECRET_KEY', 'dgr-kpi-dashboard-2024'),
        'FILES_DIRECTORY': os.environ.get('FILES_DIRECTORY', 'Files'),
        'DEMO_MODE': os.environ.get('DEMO_MODE', 'true').lower() == 'true'
    }

# Production WSGI application
def create_app():
    """Create and configure the Flask application"""
    config = get_config()
    
    app.config.update(config)
    
    # Initialize system on app creation
    with app.app_context():
        initialize_system()
    
    return app

# For Gunicorn
application = create_app()

if __name__ == '__main__':
    config = get_config()
    
    print("🚀 DGR KPI INTELLIGENCE DASHBOARD")
    print("=" * 50)
    print(f"🌐 Starting server on {config['HOST']}:{config['PORT']}")
    print(f"🔧 Debug Mode: {config['DEBUG']}")
    print(f"📊 Demo Mode: {config['DEMO_MODE']}")
    print(f"📁 Files Directory: {config['FILES_DIRECTORY']}")
    print("=" * 50)
    
    app.run(
        host=config['HOST'],
        port=config['PORT'],
        debug=config['DEBUG']
    )