"""
Advanced Analytics Engine - Day 3
KPI AI Assistant with Predictive Analytics, Trend Analysis, and Smart Reporting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import stats
import json

# Simple linear regression since sklearn might not be available
def simple_linear_regression(x, y):
    """Simple linear regression implementation"""
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate slope and intercept
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if denominator == 0:
        slope = 0
        intercept = y_mean
        r_squared = 0
    else:
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return slope, intercept, r_squared

warnings.filterwarnings('ignore')

class AdvancedAnalytics:
    """
    Advanced Analytics Engine for renewable energy plants
    Features: Trend analysis, predictions, anomaly detection, benchmarking
    """
    
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.logger = logging.getLogger(__name__)
        self.benchmark_thresholds = {
            'solar': {
                'excellent_pr': 85,      # Performance Ratio > 85%
                'good_pr': 75,
                'excellent_cuf': 25,     # Capacity Utilization > 25%
                'good_cuf': 20,
                'excellent_availability': 98,  # Availability > 98%
                'good_availability': 95
            },
            'wind': {
                'excellent_pr': 90,
                'good_pr': 80,
                'excellent_cuf': 35,
                'good_cuf': 25,
                'excellent_availability': 95,
                'good_availability': 90
            }
        }
    
    def generate_trend_analysis(self, plant_name: str, days: int = 30) -> Dict:
        """
        Analyze performance trends over specified period
        """
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            # Get plant data
            plant_data = self.data_processor.get_plant_data(plant_name)
            if plant_data is None or plant_data.empty:
                return {"error": f"No data available for {plant_name}"}
            
            # Filter by date range
            filtered_data = self.data_processor.filter_by_date_range(
                plant_data, start_date, end_date
            )
            
            if len(filtered_data) < 7:  # Need minimum data for trends
                return {"error": "Insufficient data for trend analysis"}
            
            # Clean data
            cleaned_data = self.data_processor.clean_data(filtered_data.copy())
            
            # Calculate trends
            trends = self._calculate_trends(cleaned_data)
            
            # Performance insights
            insights = self._generate_trend_insights(trends, plant_name)
            
            return {
                "plant": plant_name,
                "period": f"{days} days",
                "data_points": len(cleaned_data),
                "trends": trends,
                "insights": insights,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            
        except Exception as e:
            self.logger.error(f"Error in trend analysis for {plant_name}: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _calculate_trends(self, data: pd.DataFrame) -> Dict:
        """Calculate trend metrics for key KPIs"""
        trends = {}
        
        # Ensure we have a Date column for trend analysis
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values('Date').reset_index(drop=True)
            data['day_number'] = range(len(data))
        else:
            data['day_number'] = range(len(data))
        
        # Key metrics for trend analysis
        key_metrics = ['PR(%)', 'CUF(%)', 'PA(%)', 'EGA(%)', 'Mtr_Export (kWh)']
        
        for metric in key_metrics:
            if metric in data.columns:
                values = data[metric].dropna()
                if len(values) >= 3:  # Need minimum points for regression
                    trend_data = self._calculate_metric_trend(values, data['day_number'][:len(values)])
                    trends[metric] = trend_data
        
        return trends
    
    def _calculate_metric_trend(self, values: pd.Series, day_numbers: pd.Series) -> Dict:
        """Calculate trend statistics for a specific metric"""
        try:
            # Linear regression to find trend
            X = day_numbers.values
            y = values.values
            
            slope, intercept, r_squared = simple_linear_regression(X, y)
            
            # Statistical significance (simple correlation)
            if len(X) > 2:
                correlation = np.corrcoef(X, y)[0, 1] if not np.isnan(np.corrcoef(X, y)[0, 1]) else 0
            else:
                correlation = 0
            
            # Trend classification
            if abs(slope) < 0.1:
                trend_direction = "stable"
            elif slope > 0:
                trend_direction = "improving"
            else:
                trend_direction = "declining"
            
            # Volatility (coefficient of variation)
            volatility = (values.std() / values.mean()) * 100 if values.mean() != 0 else 0
            
            return {
                "slope": round(slope, 4),
                "direction": trend_direction,
                "r_squared": round(r_squared, 3),
                "correlation": round(correlation, 3),
                "significant": abs(correlation) > 0.3,  # Simple threshold
                "volatility": round(volatility, 2),
                "current_value": round(values.iloc[-1], 2),
                "period_avg": round(values.mean(), 2),
                "period_max": round(values.max(), 2),
                "period_min": round(values.min(), 2)
            }
            
        except Exception as e:
            self.logger.warning(f"Trend calculation failed: {str(e)}")
            return {
                "slope": 0,
                "direction": "unknown",
                "current_value": round(values.iloc[-1], 2) if len(values) > 0 else 0,
                "period_avg": round(values.mean(), 2),
                "error": str(e)
            }
    
    def _generate_trend_insights(self, trends: Dict, plant_name: str) -> List[str]:
        """Generate human-readable insights from trend data"""
        insights = []
        
        # Performance ratio insights
        if 'PR(%)' in trends:
            pr_trend = trends['PR(%)']
            if pr_trend.get('significant', False):
                if pr_trend['direction'] == 'improving':
                    insights.append(f"📈 Performance ratio is improving (+{abs(pr_trend['slope']):.2f}% per day)")
                elif pr_trend['direction'] == 'declining':
                    insights.append(f"📉 Performance ratio is declining (-{abs(pr_trend['slope']):.2f}% per day)")
            
            if pr_trend.get('volatility', 0) > 10:
                insights.append(f"⚠️ High performance volatility ({pr_trend['volatility']:.1f}%) - investigate equipment stability")
        
        # Capacity utilization insights
        if 'CUF(%)' in trends:
            cuf_trend = trends['CUF(%)']
            if cuf_trend.get('significant', False):
                if cuf_trend['direction'] == 'improving':
                    insights.append(f"🔋 Capacity utilization trending upward")
                elif cuf_trend['direction'] == 'declining':
                    insights.append(f"🔋 Capacity utilization declining - check resource availability")
        
        # Availability insights
        if 'PA(%)' in trends:
            pa_trend = trends['PA(%)']
            if pa_trend.get('current_value', 0) < 95:
                insights.append(f"🔧 Plant availability below target ({pa_trend['current_value']:.1f}%) - maintenance needed")
            elif pa_trend.get('current_value', 0) > 98:
                insights.append(f"✅ Excellent plant availability ({pa_trend['current_value']:.1f}%)")
        
        # Export energy insights
        if 'Mtr_Export (kWh)' in trends:
            export_trend = trends['Mtr_Export (kWh)']
            if export_trend.get('significant', False):
                if export_trend['direction'] == 'improving':
                    insights.append(f"💰 Energy export increasing - strong revenue performance")
                elif export_trend['direction'] == 'declining':
                    insights.append(f"💰 Energy export declining - investigate production issues")
        
        if not insights:
            insights.append("📊 Performance metrics are stable with no significant trends detected")
        
        return insights
    
    def predict_performance(self, plant_name: str, days_ahead: int = 7) -> Dict:
        """
        Predict plant performance for upcoming days
        """
        try:
            # Get historical data (last 60 days for better prediction)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=60)
            
            plant_data = self.data_processor.get_plant_data(plant_name)
            if plant_data is None or plant_data.empty:
                return {"error": f"No data available for {plant_name}"}
            
            filtered_data = self.data_processor.filter_by_date_range(
                plant_data, start_date, end_date
            )
            
            if len(filtered_data) < 14:  # Need minimum data for prediction
                return {"error": "Insufficient historical data for prediction"}
            
            cleaned_data = self.data_processor.clean_data(filtered_data.copy())
            
            # Generate predictions
            predictions = self._generate_predictions(cleaned_data, days_ahead)
            
            # Confidence assessment
            confidence = self._assess_prediction_confidence(cleaned_data)
            
            return {
                "plant": plant_name,
                "prediction_period": f"{days_ahead} days",
                "historical_data_points": len(cleaned_data),
                "predictions": predictions,
                "confidence": confidence,
                "forecast_date": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            
        except Exception as e:
            self.logger.error(f"Error in prediction for {plant_name}: {str(e)}")
            return {"error": f"Prediction failed: {str(e)}"}
    
    def _generate_predictions(self, data: pd.DataFrame, days_ahead: int) -> Dict:
        """Generate predictions using simple linear regression"""
        predictions = {}
        
        if len(data) < 7:
            return {"error": "Insufficient data for prediction"}
        
        # Prepare time series data
        data['day_number'] = range(len(data))
        
        # Key metrics to predict
        key_metrics = ['PR(%)', 'CUF(%)', 'Mtr_Export (kWh)']
        
        for metric in key_metrics:
            if metric in data.columns:
                values = data[metric].dropna()
                if len(values) >= 7:
                    pred_result = self._predict_metric(values, data['day_number'][:len(values)], days_ahead)
                    predictions[metric] = pred_result
        
        return predictions
    
    def _predict_metric(self, values: pd.Series, day_numbers: pd.Series, days_ahead: int) -> Dict:
        """Predict future values for a specific metric"""
        try:
            # Fit linear regression
            X = day_numbers.values
            y = values.values
            
            slope, intercept, r_squared = simple_linear_regression(X, y)
            
            # Predict future values
            future_days = np.arange(len(values), len(values) + days_ahead)
            predictions = slope * future_days + intercept
            
            # Calculate prediction intervals (simple approach)
            residuals = y - (slope * X + intercept)
            mse = np.mean(residuals ** 2)
            std_error = np.sqrt(mse)
            
            # Generate forecast
            forecast = []
            for i, pred in enumerate(predictions):
                forecast_date = datetime.now().date() + timedelta(days=i+1)
                forecast.append({
                    "date": forecast_date.strftime("%Y-%m-%d"),
                    "predicted_value": round(max(0, pred), 2),  # Ensure non-negative
                    "confidence_interval": {
                        "lower": round(max(0, pred - 1.96 * std_error), 2),
                        "upper": round(pred + 1.96 * std_error, 2)
                    }
                })
            
            return {
                "forecast": forecast,
                "model_r_squared": round(r_squared, 3),
                "trend_slope": round(slope, 4),
                "recent_avg": round(values.tail(7).mean(), 2)
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def _assess_prediction_confidence(self, data: pd.DataFrame) -> Dict:
        """Assess confidence in predictions based on data quality"""
        
        # Data completeness
        completeness = (data.notna().sum().sum() / (len(data) * len(data.columns))) * 100
        
        # Data consistency (coefficient of variation for key metrics)
        consistency_scores = []
        key_metrics = ['PR(%)', 'CUF(%)', 'PA(%)']
        
        for metric in key_metrics:
            if metric in data.columns:
                values = data[metric].dropna()
                if len(values) > 0 and values.mean() != 0:
                    cv = (values.std() / values.mean()) * 100
                    consistency_scores.append(min(100, max(0, 100 - cv)))  # Higher is better
        
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 50
        
        # Overall confidence score
        confidence_score = (completeness * 0.4 + avg_consistency * 0.4 + min(100, len(data) * 2) * 0.2)
        
        # Confidence level
        if confidence_score >= 80:
            level = "High"
        elif confidence_score >= 60:
            level = "Medium"
        else:
            level = "Low"
        
        return {
            "score": round(confidence_score, 1),
            "level": level,
            "data_completeness": round(completeness, 1),
            "data_consistency": round(avg_consistency, 1),
            "historical_days": len(data)
        }
    
    def benchmark_performance(self, plant_name: str, plant_type: str = "solar") -> Dict:
        """
        Benchmark plant performance against industry standards
        """
        try:
            # Get recent performance (last 30 days)
            summary = self.data_processor.get_plant_summary(plant_name, days=30)
            
            if not summary or summary.get('total_days', 0) == 0:
                return {"error": f"No recent data for {plant_name}"}
            
            # Get benchmarks for plant type
            benchmarks = self.benchmark_thresholds.get(plant_type, self.benchmark_thresholds['solar'])
            
            # Calculate benchmark scores
            scores = self._calculate_benchmark_scores(summary, benchmarks)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(scores, plant_name, plant_type)
            
            return {
                "plant": plant_name,
                "plant_type": plant_type,
                "benchmark_period": "30 days",
                "performance_scores": scores,
                "overall_grade": self._calculate_overall_grade(scores),
                "recommendations": recommendations,
                "benchmark_date": datetime.now().strftime("%Y-%m-%d")
            }
            
        except Exception as e:
            self.logger.error(f"Error in benchmarking for {plant_name}: {str(e)}")
            return {"error": f"Benchmarking failed: {str(e)}"}
    
    def _calculate_benchmark_scores(self, summary: Dict, benchmarks: Dict) -> Dict:
        """Calculate performance scores against benchmarks"""
        scores = {}
        
        # Performance Ratio scoring
        pr = summary.get('avg_performance_ratio', 0)
        if pr >= benchmarks['excellent_pr']:
            pr_score = 100
            pr_grade = "Excellent"
        elif pr >= benchmarks['good_pr']:
            pr_score = 80
            pr_grade = "Good"
        elif pr >= benchmarks['good_pr'] - 10:
            pr_score = 60
            pr_grade = "Fair"
        else:
            pr_score = 40
            pr_grade = "Poor"
        
        scores['performance_ratio'] = {
            "value": round(pr, 2),
            "score": pr_score,
            "grade": pr_grade,
            "benchmark": benchmarks['excellent_pr']
        }
        
        # Capacity Utilization scoring
        cuf = summary.get('avg_plf', 0)  # Using PLF as proxy for CUF
        if cuf >= benchmarks['excellent_cuf']:
            cuf_score = 100
            cuf_grade = "Excellent"
        elif cuf >= benchmarks['good_cuf']:
            cuf_score = 80
            cuf_grade = "Good"
        elif cuf >= benchmarks['good_cuf'] - 5:
            cuf_score = 60
            cuf_grade = "Fair"
        else:
            cuf_score = 40
            cuf_grade = "Poor"
        
        scores['capacity_utilization'] = {
            "value": round(cuf, 2),
            "score": cuf_score,
            "grade": cuf_grade,
            "benchmark": benchmarks['excellent_cuf']
        }
        
        # Availability scoring
        availability = summary.get('avg_availability', 0)
        if availability >= benchmarks['excellent_availability']:
            avail_score = 100
            avail_grade = "Excellent"
        elif availability >= benchmarks['good_availability']:
            avail_score = 80
            avail_grade = "Good"
        elif availability >= benchmarks['good_availability'] - 5:
            avail_score = 60
            avail_grade = "Fair"
        else:
            avail_score = 40
            avail_grade = "Poor"
        
        scores['availability'] = {
            "value": round(availability, 2),
            "score": avail_score,
            "grade": avail_grade,
            "benchmark": benchmarks['excellent_availability']
        }
        
        return scores
    
    def _calculate_overall_grade(self, scores: Dict) -> Dict:
        """Calculate overall performance grade"""
        total_score = np.mean([scores[metric]["score"] for metric in scores])
        
        if total_score >= 90:
            grade = "A+"
            description = "Outstanding Performance"
        elif total_score >= 80:
            grade = "A"
            description = "Excellent Performance"
        elif total_score >= 70:
            grade = "B"
            description = "Good Performance"
        elif total_score >= 60:
            grade = "C"
            description = "Fair Performance"
        else:
            grade = "D"
            description = "Needs Improvement"
        
        return {
            "score": round(total_score, 1),
            "grade": grade,
            "description": description
        }
    
    def _generate_recommendations(self, scores: Dict, plant_name: str, plant_type: str) -> List[str]:
        """Generate actionable recommendations based on performance scores"""
        recommendations = []
        
        # Performance Ratio recommendations
        pr_score = scores.get('performance_ratio', {}).get('score', 0)
        if pr_score < 70:
            recommendations.append("🔧 Performance Ratio below target - check inverter efficiency and module soiling")
            recommendations.append("📊 Analyze string-level performance to identify underperforming sections")
        
        # Capacity Utilization recommendations
        cuf_score = scores.get('capacity_utilization', {}).get('score', 0)
        if cuf_score < 70:
            recommendations.append("⚡ Low capacity utilization - review resource assessment and system design")
            if plant_type == "solar":
                recommendations.append("☀️ Consider module cleaning schedule and tracking system optimization")
            else:
                recommendations.append("💨 Review wind turbine blade condition and pitch control settings")
        
        # Availability recommendations
        avail_score = scores.get('availability', {}).get('score', 0)
        if avail_score < 80:
            recommendations.append("🔧 Low availability - implement preventive maintenance program")
            recommendations.append("📈 Establish equipment monitoring and early warning systems")
        
        # Overall recommendations
        overall_score = np.mean([scores[metric]["score"] for metric in scores])
        if overall_score >= 85:
            recommendations.append("🏆 Excellent performance - consider this plant as a benchmark for others")
        elif overall_score < 60:
            recommendations.append("⚠️ Comprehensive performance review recommended")
            recommendations.append("💼 Consider engaging O&M optimization consultant")
        
        return recommendations if recommendations else ["✅ Plant performance meets industry standards"]
    
    def detect_anomalies(self, plant_name: str, days: int = 30) -> Dict:
        """
        Detect performance anomalies using statistical methods
        """
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            plant_data = self.data_processor.get_plant_data(plant_name)
            if plant_data is None or plant_data.empty:
                return {"error": f"No data available for {plant_name}"}
            
            filtered_data = self.data_processor.filter_by_date_range(
                plant_data, start_date, end_date
            )
            
            if len(filtered_data) < 7:
                return {"error": "Insufficient data for anomaly detection"}
            
            cleaned_data = self.data_processor.clean_data(filtered_data.copy())
            
            # Detect anomalies
            anomalies = self._detect_statistical_anomalies(cleaned_data)
            
            # Classify anomaly severity
            severity_analysis = self._classify_anomaly_severity(anomalies)
            
            return {
                "plant": plant_name,
                "analysis_period": f"{days} days",
                "total_data_points": len(cleaned_data),
                "anomalies_detected": anomalies,
                "severity_analysis": severity_analysis,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection for {plant_name}: {str(e)}")
            return {"error": f"Anomaly detection failed: {str(e)}"}
    
    def _detect_statistical_anomalies(self, data: pd.DataFrame) -> Dict:
        """Detect anomalies using Z-score and IQR methods"""
        anomalies = {}
        
        key_metrics = ['PR(%)', 'CUF(%)', 'PA(%)', 'Mtr_Export (kWh)']
        
        for metric in key_metrics:
            if metric in data.columns:
                values = data[metric].dropna()
                if len(values) >= 7:
                    metric_anomalies = []
                    
                    # Z-score method (values > 3 standard deviations)
                    if values.std() > 0:
                        z_scores = np.abs((values - values.mean()) / values.std())
                        z_anomalies = values[z_scores > 3]
                    else:
                        z_anomalies = pd.Series(dtype=float)
                    
                    # IQR method
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1
                    iqr_anomalies = values[(values < (Q1 - 2.5 * IQR)) | (values > (Q3 + 2.5 * IQR))]
                    
                    # Combine and format anomalies
                    all_anomalies = pd.concat([z_anomalies, iqr_anomalies]).drop_duplicates()
                    
                    for idx, value in all_anomalies.items():
                        if 'Date' in data.columns:
                            date = data.loc[idx, 'Date'] if idx in data.index else "Unknown"
                        else:
                            date = f"Day {idx + 1}"
                        
                        z_score = abs((value - values.mean()) / values.std()) if values.std() > 0 else 0
                        severity = "high" if z_score > 4 else "medium"
                        
                        metric_anomalies.append({
                            "date": str(date),
                            "value": round(value, 2),
                            "expected_range": f"{round(Q1, 2)} - {round(Q3, 2)}",
                            "severity": severity
                        })
                    
                    if metric_anomalies:
                        anomalies[metric] = {
                            "count": len(metric_anomalies),
                            "percentage": round((len(metric_anomalies) / len(values)) * 100, 2),
                            "details": metric_anomalies[:5]  # Show top 5 anomalies
                        }
        
        return anomalies
    
    def _classify_anomaly_severity(self, anomalies: Dict) -> Dict:
        """Classify overall anomaly severity and impact"""
        total_anomalies = sum([anomalies[metric]["count"] for metric in anomalies])
        
        if total_anomalies == 0:
            severity = "None"
            impact = "No anomalies detected - plant operating normally"
        elif total_anomalies <= 3:
            severity = "Low"
            impact = "Minor irregularities - monitor closely"
        elif total_anomalies <= 7:
            severity = "Medium"
            impact = "Moderate anomalies - investigate potential issues"
        else:
            severity = "High"
            impact = "Multiple anomalies detected - immediate attention required"
        
        # Identify most problematic metrics
        problem_metrics = []
        for metric, data in anomalies.items():
            if data["percentage"] > 10:  # More than 10% anomalous values
                problem_metrics.append(metric)
        
        return {
            "overall_severity": severity,
            "impact_assessment": impact,
            "total_anomalies": total_anomalies,
            "problem_metrics": problem_metrics,
            "recommendation": self._get_anomaly_recommendation(severity, problem_metrics)
        }
    
    def _get_anomaly_recommendation(self, severity: str, problem_metrics: List[str]) -> str:
        """Get recommendation based on anomaly analysis"""
        if severity == "None":
            return "Continue normal operations and monitoring"
        elif severity == "Low":
            return "Increase monitoring frequency and review recent maintenance activities"
        elif severity == "Medium":
            return "Investigate equipment performance and consider diagnostic testing"
        else:
            recommendations = ["Immediate investigation required"]
            if "PR(%)" in problem_metrics:
                recommendations.append("Check inverter performance and module conditions")
            if "PA(%)" in problem_metrics:
                recommendations.append("Review equipment availability and maintenance logs")
            if "Mtr_Export (kWh)" in problem_metrics:
                recommendations.append("Investigate grid connection and metering systems")
            
            return " | ".join(recommendations)