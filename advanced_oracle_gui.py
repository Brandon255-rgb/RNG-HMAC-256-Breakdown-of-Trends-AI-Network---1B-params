#!/usr/bin/env python3
"""
ADVANCED ORACLE GUI
===================
Modern dark-mode interface with:
- Live roll graph with matplotlib
- Confidence meter
- Bias indicator (-0.19 under/over)
- Demo/Real toggle
- Auto-bet switch
- Real-time P&L tracker
- Advanced weighted decisions for Stake API betting

Built with customtkinter and matplotlib embedded
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import threading
import time
import asyncio
import json
import requests
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Any, Optional
import os
from dotenv import load_dotenv

# Import our systems
from bedrock_ai_brain import BedrockAIBrain, BettingContext, AIDecision
from demo_analysis_system import DemoAnalysisSystem
from supreme_bedrock_bot import SupremeBedrockBot, BettingDecision
from oracle_support_utils import detect_streaks, shannon_entropy

load_dotenv()

class AdvancedOracleGUI:
    """Advanced Oracle GUI with modern dark theme"""
    
    def __init__(self):
        # Set appearance mode and color theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("🔮 Advanced Oracle AI - Supreme Betting System")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)
        
        # Oracle systems
        self.demo_analyzer = None
        self.bedrock_bot = None
        self.ai_brain = None
        
        # Data tracking
        self.live_rolls = deque(maxlen=100)
        self.predictions = deque(maxlen=50)
        self.betting_history = []
        self.pnl_history = deque(maxlen=100)
        
        # State variables
        self.is_demo_mode = True
        self.auto_betting = False
        self.current_balance = 1000.0
        self.session_pnl = 0.0
        self.confidence_score = 0.0
        self.bias_indicator = 0.0
        self.oracle_active = False
        
        # Setup GUI
        self.setup_gui()
        self.setup_matplotlib()
        self.initialize_systems()
        
        # Start update loop
        self.start_update_loop()
    
    def setup_gui(self):
        """Setup the main GUI layout"""
        
        # Configure grid
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=2)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Create main frames
        self.left_frame = ctk.CTkFrame(self.root, corner_radius=15)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        
        self.right_frame = ctk.CTkFrame(self.root, corner_radius=15)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        
        self.setup_left_panel()
        self.setup_right_panel()
    
    def setup_left_panel(self):
        """Setup left control panel"""
        
        # Title
        title_label = ctk.CTkLabel(
            self.left_frame,
            text="🔮 Advanced Oracle",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(20, 10))
        
        # Mode Toggle Section
        mode_frame = ctk.CTkFrame(self.left_frame, corner_radius=10)
        mode_frame.pack(fill="x", padx=20, pady=10)
        
        mode_label = ctk.CTkLabel(mode_frame, text="Trading Mode", font=ctk.CTkFont(size=16, weight="bold"))
        mode_label.pack(pady=(10, 5))
        
        self.mode_switch = ctk.CTkSwitch(
            mode_frame,
            text="Demo Mode",
            command=self.toggle_mode,
            font=ctk.CTkFont(size=14)
        )
        self.mode_switch.pack(pady=(5, 10))
        self.mode_switch.select()  # Start in demo mode
        
        self.mode_status = ctk.CTkLabel(
            mode_frame,
            text="🛡️ Demo Mode - Safe Testing",
            font=ctk.CTkFont(size=12),
            text_color="#10B981"
        )
        self.mode_status.pack(pady=(0, 10))
        
        # Oracle Stats
        stats_frame = ctk.CTkFrame(self.left_frame, corner_radius=10)
        stats_frame.pack(fill="x", padx=20, pady=10)
        
        stats_label = ctk.CTkLabel(stats_frame, text="Oracle Statistics", font=ctk.CTkFont(size=16, weight="bold"))
        stats_label.pack(pady=(10, 5))
        
        # Confidence Meter
        conf_frame = ctk.CTkFrame(stats_frame, corner_radius=8)
        conf_frame.pack(fill="x", padx=10, pady=5)
        
        conf_label = ctk.CTkLabel(conf_frame, text="Confidence Level", font=ctk.CTkFont(size=12))
        conf_label.pack(pady=(5, 0))
        
        self.confidence_progress = ctk.CTkProgressBar(conf_frame, width=250, height=20)
        self.confidence_progress.pack(pady=5)
        self.confidence_progress.set(0)
        
        self.confidence_label = ctk.CTkLabel(conf_frame, text="0%", font=ctk.CTkFont(size=14, weight="bold"))
        self.confidence_label.pack(pady=(0, 5))
        
        # Bias Indicator
        bias_frame = ctk.CTkFrame(stats_frame, corner_radius=8)
        bias_frame.pack(fill="x", padx=10, pady=5)
        
        bias_label = ctk.CTkLabel(bias_frame, text="Bias Indicator", font=ctk.CTkFont(size=12))
        bias_label.pack(pady=(5, 0))
        
        self.bias_label = ctk.CTkLabel(
            bias_frame,
            text="0.00 (Neutral)",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#6B7280"
        )
        self.bias_label.pack(pady=(0, 5))
        
        # P&L Tracker
        pnl_frame = ctk.CTkFrame(stats_frame, corner_radius=8)
        pnl_frame.pack(fill="x", padx=10, pady=5)
        
        pnl_title = ctk.CTkLabel(pnl_frame, text="Session P&L", font=ctk.CTkFont(size=12))
        pnl_title.pack(pady=(5, 0))
        
        self.pnl_label = ctk.CTkLabel(
            pnl_frame,
            text="$0.00",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#10B981"
        )
        self.pnl_label.pack(pady=(0, 5))
        
        stats_frame.pack_configure(pady=(10, 15))
        
        # Control Buttons
        controls_frame = ctk.CTkFrame(self.left_frame, corner_radius=10)
        controls_frame.pack(fill="x", padx=20, pady=10)
        
        controls_label = ctk.CTkLabel(controls_frame, text="Oracle Controls", font=ctk.CTkFont(size=16, weight="bold"))
        controls_label.pack(pady=(10, 5))
        
        # Demo Analysis Button
        self.demo_btn = ctk.CTkButton(
            controls_frame,
            text="🔍 Start Demo Analysis",
            command=self.start_demo_analysis,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#8B5CF6",
            hover_color="#7C3AED"
        )
        self.demo_btn.pack(pady=5, padx=10, fill="x")
        
        # Auto Betting Toggle
        self.auto_btn = ctk.CTkButton(
            controls_frame,
            text="🤖 Start Auto Betting",
            command=self.toggle_auto_betting,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#10B981",
            hover_color="#059669",
            state="disabled"
        )
        self.auto_btn.pack(pady=5, padx=10, fill="x")
        
        # Manual Bet Controls
        manual_frame = ctk.CTkFrame(controls_frame, corner_radius=8)
        manual_frame.pack(fill="x", padx=10, pady=10)
        
        manual_label = ctk.CTkLabel(manual_frame, text="Manual Betting", font=ctk.CTkFont(size=12))
        manual_label.pack(pady=(5, 0))
        
        # Bet amount
        bet_frame = ctk.CTkFrame(manual_frame, corner_radius=5)
        bet_frame.pack(fill="x", padx=10, pady=5)
        
        bet_label = ctk.CTkLabel(bet_frame, text="Bet Amount:", font=ctk.CTkFont(size=11))
        bet_label.pack(side="left", padx=(10, 5))
        
        self.bet_entry = ctk.CTkEntry(bet_frame, width=100, placeholder_text="1.00")
        self.bet_entry.pack(side="right", padx=(5, 10))
        
        # Target value
        target_frame = ctk.CTkFrame(manual_frame, corner_radius=5)
        target_frame.pack(fill="x", padx=10, pady=5)
        
        target_label = ctk.CTkLabel(target_frame, text="Target:", font=ctk.CTkFont(size=11))
        target_label.pack(side="left", padx=(10, 5))
        
        self.target_entry = ctk.CTkEntry(target_frame, width=100, placeholder_text="50.00")
        self.target_entry.pack(side="right", padx=(5, 10))
        
        # Manual bet button
        self.manual_bet_btn = ctk.CTkButton(
            manual_frame,
            text="Place Manual Bet",
            command=self.place_manual_bet,
            height=35,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.manual_bet_btn.pack(pady=(5, 10), padx=10, fill="x")
        
        controls_frame.pack_configure(pady=(10, 20))
    
    def setup_right_panel(self):
        """Setup right panel with graphs and data"""
        
        # Create notebook for tabs
        self.notebook = ctk.CTkTabview(self.right_frame, corner_radius=10)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Create tabs
        self.notebook.add("Live Rolls")
        self.notebook.add("Predictions")
        self.notebook.add("P&L History")
        
        self.setup_live_rolls_tab()
        self.setup_predictions_tab()
        self.setup_pnl_tab()
    
    def setup_live_rolls_tab(self):
        """Setup live rolls graph tab"""
        
        tab = self.notebook.tab("Live Rolls")
        
        # Graph frame
        graph_frame = ctk.CTkFrame(tab, corner_radius=10)
        graph_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create matplotlib figure
        self.rolls_fig = Figure(figsize=(10, 6), facecolor='#2B2B2B')
        self.rolls_ax = self.rolls_fig.add_subplot(111)
        self.rolls_ax.set_facecolor('#1E1E1E')
        self.rolls_ax.set_title('Live Roll Results', color='white', fontsize=14, weight='bold')
        self.rolls_ax.set_xlabel('Roll Number', color='white')
        self.rolls_ax.set_ylabel('Result Value', color='white')
        self.rolls_ax.tick_params(colors='white')
        self.rolls_ax.grid(True, alpha=0.3, color='gray')
        
        # Add reference lines
        self.rolls_ax.axhline(y=50, color='yellow', linestyle='--', alpha=0.7, label='50% Line')
        self.rolls_ax.axhline(y=25, color='red', linestyle=':', alpha=0.5, label='Low Zone')
        self.rolls_ax.axhline(y=75, color='green', linestyle=':', alpha=0.5, label='High Zone')
        self.rolls_ax.legend(facecolor='#2B2B2B', edgecolor='white', labelcolor='white')
        
        # Canvas
        self.rolls_canvas = FigureCanvasTkAgg(self.rolls_fig, graph_frame)
        self.rolls_canvas.draw()
        self.rolls_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        # Stats below graph
        stats_frame = ctk.CTkFrame(tab, corner_radius=8)
        stats_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        stats_grid = ctk.CTkFrame(stats_frame, corner_radius=5)
        stats_grid.pack(fill="x", padx=10, pady=10)
        
        # Configure grid
        for i in range(4):
            stats_grid.grid_columnconfigure(i, weight=1)
        
        # Recent stats
        self.avg_label = ctk.CTkLabel(stats_grid, text="Avg: --", font=ctk.CTkFont(size=12, weight="bold"))
        self.avg_label.grid(row=0, column=0, padx=5, pady=5)
        
        self.std_label = ctk.CTkLabel(stats_grid, text="Std: --", font=ctk.CTkFont(size=12, weight="bold"))
        self.std_label.grid(row=0, column=1, padx=5, pady=5)
        
        self.streak_label = ctk.CTkLabel(stats_grid, text="Streak: --", font=ctk.CTkFont(size=12, weight="bold"))
        self.streak_label.grid(row=0, column=2, padx=5, pady=5)
        
        self.entropy_label = ctk.CTkLabel(stats_grid, text="Entropy: --", font=ctk.CTkFont(size=12, weight="bold"))
        self.entropy_label.grid(row=0, column=3, padx=5, pady=5)
    
    def setup_predictions_tab(self):
        """Setup predictions graph tab"""
        
        tab = self.notebook.tab("Predictions")
        
        # Prediction accuracy graph
        graph_frame = ctk.CTkFrame(tab, corner_radius=10)
        graph_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.pred_fig = Figure(figsize=(10, 6), facecolor='#2B2B2B')
        self.pred_ax = self.pred_fig.add_subplot(111)
        self.pred_ax.set_facecolor('#1E1E1E')
        self.pred_ax.set_title('Prediction Accuracy', color='white', fontsize=14, weight='bold')
        self.pred_ax.set_xlabel('Prediction Number', color='white')
        self.pred_ax.set_ylabel('Accuracy %', color='white')
        self.pred_ax.tick_params(colors='white')
        self.pred_ax.grid(True, alpha=0.3, color='gray')
        
        self.pred_canvas = FigureCanvasTkAgg(self.pred_fig, graph_frame)
        self.pred_canvas.draw()
        self.pred_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        # Prediction controls
        pred_controls = ctk.CTkFrame(tab, corner_radius=8)
        pred_controls.pack(fill="x", padx=10, pady=(0, 10))
        
        pred_label = ctk.CTkLabel(pred_controls, text="Prediction Settings", font=ctk.CTkFont(size=14, weight="bold"))
        pred_label.pack(pady=(10, 5))
        
        # Confidence threshold
        threshold_frame = ctk.CTkFrame(pred_controls, corner_radius=5)
        threshold_frame.pack(fill="x", padx=10, pady=5)
        
        threshold_label = ctk.CTkLabel(threshold_frame, text="Min Confidence:", font=ctk.CTkFont(size=12))
        threshold_label.pack(side="left", padx=(10, 5))
        
        self.confidence_threshold = ctk.CTkSlider(
            threshold_frame,
            from_=50,
            to=95,
            number_of_steps=45,
            command=self.update_threshold
        )
        self.confidence_threshold.pack(side="left", expand=True, fill="x", padx=10)
        self.confidence_threshold.set(70)
        
        self.threshold_label = ctk.CTkLabel(threshold_frame, text="70%", font=ctk.CTkFont(size=12, weight="bold"))
        self.threshold_label.pack(side="right", padx=(5, 10))
    
    def setup_pnl_tab(self):
        """Setup P&L history tab"""
        
        tab = self.notebook.tab("P&L History")
        
        # P&L graph
        graph_frame = ctk.CTkFrame(tab, corner_radius=10)
        graph_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.pnl_fig = Figure(figsize=(10, 6), facecolor='#2B2B2B')
        self.pnl_ax = self.pnl_fig.add_subplot(111)
        self.pnl_ax.set_facecolor('#1E1E1E')
        self.pnl_ax.set_title('Profit & Loss History', color='white', fontsize=14, weight='bold')
        self.pnl_ax.set_xlabel('Time', color='white')
        self.pnl_ax.set_ylabel('Cumulative P&L ($)', color='white')
        self.pnl_ax.tick_params(colors='white')
        self.pnl_ax.grid(True, alpha=0.3, color='gray')
        
        # Zero line
        self.pnl_ax.axhline(y=0, color='white', linestyle='-', alpha=0.8)
        
        self.pnl_canvas = FigureCanvasTkAgg(self.pnl_fig, graph_frame)
        self.pnl_canvas.draw()
        self.pnl_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
        # P&L summary
        summary_frame = ctk.CTkFrame(tab, corner_radius=8)
        summary_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        summary_label = ctk.CTkLabel(summary_frame, text="Session Summary", font=ctk.CTkFont(size=14, weight="bold"))
        summary_label.pack(pady=(10, 5))
        
        summary_grid = ctk.CTkFrame(summary_frame, corner_radius=5)
        summary_grid.pack(fill="x", padx=10, pady=10)
        
        for i in range(3):
            summary_grid.grid_columnconfigure(i, weight=1)
        
        self.total_bets_label = ctk.CTkLabel(summary_grid, text="Bets: 0", font=ctk.CTkFont(size=12))
        self.total_bets_label.grid(row=0, column=0, padx=5, pady=5)
        
        self.win_rate_label = ctk.CTkLabel(summary_grid, text="Win Rate: 0%", font=ctk.CTkFont(size=12))
        self.win_rate_label.grid(row=0, column=1, padx=5, pady=5)
        
        self.roi_label = ctk.CTkLabel(summary_grid, text="ROI: 0%", font=ctk.CTkFont(size=12))
        self.roi_label.grid(row=0, column=2, padx=5, pady=5)
    
    def setup_matplotlib(self):
        """Configure matplotlib for dark theme"""
        plt.style.use('dark_background')
    
    def initialize_systems(self):
        """Initialize Oracle systems"""
        try:
            stake_api_key = os.getenv('STAKE_API_KEY', '')
            
            self.demo_analyzer = DemoAnalysisSystem(stake_api_key)
            self.bedrock_bot = SupremeBedrockBot()
            self.ai_brain = BedrockAIBrain()
            
            print("✅ Oracle systems initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize systems: {e}")
            messagebox.showerror("Initialization Error", f"Failed to initialize Oracle systems:\n{str(e)}")
    
    def toggle_mode(self):
        """Toggle between demo and real mode"""
        self.is_demo_mode = self.mode_switch.get()
        
        if self.is_demo_mode:
            self.mode_status.configure(
                text="🛡️ Demo Mode - Safe Testing",
                text_color="#10B981"
            )
        else:
            self.mode_status.configure(
                text="⚠️ REAL MODE - Live Trading",
                text_color="#EF4444"
            )
    
    def start_demo_analysis(self):
        """Start demo analysis"""
        if self.demo_analyzer is None:
            messagebox.showerror("Error", "Demo analyzer not initialized")
            return
        
        # Disable button and show progress
        self.demo_btn.configure(text="🔄 Analyzing...", state="disabled")
        
        def run_analysis():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.demo_analyzer.start_demo_analysis())
                
                # Update GUI from main thread
                self.root.after(0, self.analysis_complete, result)
                
            except Exception as e:
                self.root.after(0, self.analysis_error, str(e))
        
        # Run analysis in background thread
        analysis_thread = threading.Thread(target=run_analysis, daemon=True)
        analysis_thread.start()
    
    def analysis_complete(self, result):
        """Handle analysis completion"""
        
        self.demo_btn.configure(text="✅ Analysis Complete", fg_color="#10B981")
        self.auto_btn.configure(state="normal")
        
        # Update confidence and bias
        if 'confidence' in result:
            self.confidence_score = result['confidence']
            self.update_confidence_display()
        
        if 'seed_analysis' in result and 'bias_detected' in result['seed_analysis']:
            self.bias_indicator = result['seed_analysis']['bias_detected']
            self.update_bias_display()
        
        messagebox.showinfo("Success", "Demo analysis complete! Oracle is ready for auto betting.")
        
    def analysis_error(self, error):
        """Handle analysis error"""
        self.demo_btn.configure(text="❌ Analysis Failed", fg_color="#EF4444")
        messagebox.showerror("Analysis Error", f"Demo analysis failed:\n{error}")
    
    def toggle_auto_betting(self):
        """Toggle auto betting"""
        self.auto_betting = not self.auto_betting
        
        if self.auto_betting:
            self.auto_btn.configure(
                text="🛑 Stop Auto Betting",
                fg_color="#EF4444",
                hover_color="#DC2626"
            )
            self.start_auto_betting()
        else:
            self.auto_btn.configure(
                text="🤖 Start Auto Betting",
                fg_color="#10B981",
                hover_color="#059669"
            )
            self.stop_auto_betting()
    
    def start_auto_betting(self):
        """Start auto betting loop"""
        print("🤖 Auto betting started")
        
        def auto_bet_loop():
            while self.auto_betting:
                try:
                    # Generate prediction and place bet
                    if self.confidence_score > 60:  # Only bet if confident
                        self.place_auto_bet()
                    
                    # Wait before next bet
                    time.sleep(10)  # 10 seconds between bets
                    
                except Exception as e:
                    print(f"❌ Auto betting error: {e}")
                    break
        
        auto_thread = threading.Thread(target=auto_bet_loop, daemon=True)
        auto_thread.start()
    
    def stop_auto_betting(self):
        """Stop auto betting"""
        print("🛑 Auto betting stopped")
    
    def place_manual_bet(self):
        """Place manual bet"""
        try:
            bet_amount = float(self.bet_entry.get() or "1.0")
            target = float(self.target_entry.get() or "50.0")
            
            # Simulate bet result
            result = np.random.uniform(0, 100)
            win = result < target
            
            # Update P&L
            if win:
                multiplier = 99 / target
                profit = bet_amount * (multiplier - 1)
                color = "#10B981"
            else:
                profit = -bet_amount
                color = "#EF4444"
            
            self.session_pnl += profit
            self.pnl_history.append(self.session_pnl)
            
            # Update displays
            self.update_pnl_display()
            self.update_graphs()
            
            # Add to betting history
            self.betting_history.append({
                'time': datetime.now(),
                'bet_amount': bet_amount,
                'target': target,
                'result': result,
                'win': win,
                'profit': profit
            })
            
            # Show result
            result_msg = f"Result: {result:.2f}\n{'WIN' if win else 'LOSS'}: ${profit:+.2f}"
            messagebox.showinfo("Bet Result", result_msg)
            
        except ValueError:
            messagebox.showerror("Error", "Invalid bet amount or target")
        except Exception as e:
            messagebox.showerror("Error", f"Betting error: {str(e)}")
    
    def place_auto_bet(self):
        """Place automatic bet based on AI prediction"""
        # This would integrate with real betting logic
        # For demo, simulate a bet
        
        bet_amount = 1.0  # Default bet amount
        
        # Use bias indicator to determine bet
        if abs(self.bias_indicator) > 0.1:
            target = 45.0 if self.bias_indicator < 0 else 55.0
        else:
            target = 50.0
        
        # Simulate result
        result = np.random.uniform(0, 100)
        
        # Add to live rolls
        self.live_rolls.append(result)
        
        # Calculate win/loss
        win = result < target if self.bias_indicator < 0 else result > target
        
        if win:
            multiplier = 99 / (target if target < 50 else (100 - target))
            profit = bet_amount * (multiplier - 1)
        else:
            profit = -bet_amount
        
        self.session_pnl += profit
        self.pnl_history.append(self.session_pnl)
        
        # Update displays
        self.root.after(0, self.update_displays)
        
        print(f"Auto bet: {bet_amount} @ {target:.1f}, Result: {result:.2f}, P&L: {profit:+.2f}")
    
    def update_displays(self):
        """Update all displays"""
        self.update_confidence_display()
        self.update_bias_display()
        self.update_pnl_display()
        self.update_graphs()
    
    def update_confidence_display(self):
        """Update confidence meter"""
        confidence_pct = self.confidence_score / 100.0
        self.confidence_progress.set(confidence_pct)
        self.confidence_label.configure(text=f"{self.confidence_score:.0f}%")
        
        # Color based on confidence level
        if self.confidence_score >= 80:
            color = "#10B981"  # Green
        elif self.confidence_score >= 60:
            color = "#F59E0B"  # Yellow
        else:
            color = "#EF4444"  # Red
        
        self.confidence_label.configure(text_color=color)
    
    def update_bias_display(self):
        """Update bias indicator"""
        if abs(self.bias_indicator) < 0.05:
            bias_text = f"{self.bias_indicator:.3f} (Neutral)"
            color = "#6B7280"
        elif self.bias_indicator < 0:
            bias_text = f"{self.bias_indicator:.3f} (Under Bias)"
            color = "#EF4444"
        else:
            bias_text = f"{self.bias_indicator:.3f} (Over Bias)"
            color = "#10B981"
        
        self.bias_label.configure(text=bias_text, text_color=color)
    
    def update_pnl_display(self):
        """Update P&L display"""
        color = "#10B981" if self.session_pnl >= 0 else "#EF4444"
        self.pnl_label.configure(text=f"${self.session_pnl:+.2f}", text_color=color)
        
        # Update summary stats
        if self.betting_history:
            total_bets = len(self.betting_history)
            wins = sum(1 for bet in self.betting_history if bet['win'])
            win_rate = (wins / total_bets) * 100
            
            initial_balance = 1000.0
            roi = (self.session_pnl / initial_balance) * 100
            
            self.total_bets_label.configure(text=f"Bets: {total_bets}")
            self.win_rate_label.configure(text=f"Win Rate: {win_rate:.1f}%")
            self.roi_label.configure(text=f"ROI: {roi:+.1f}%")
    
    def update_graphs(self):
        """Update all graphs"""
        self.update_rolls_graph()
        self.update_predictions_graph()
        self.update_pnl_graph()
    
    def update_rolls_graph(self):
        """Update live rolls graph"""
        if not self.live_rolls:
            return
        
        self.rolls_ax.clear()
        
        # Plot data
        x = list(range(len(self.live_rolls)))
        y = list(self.live_rolls)
        
        self.rolls_ax.plot(x, y, color='#3B82F6', linewidth=2, marker='o', markersize=4)
        
        # Add reference lines
        self.rolls_ax.axhline(y=50, color='yellow', linestyle='--', alpha=0.7, label='50% Line')
        self.rolls_ax.axhline(y=25, color='red', linestyle=':', alpha=0.5, label='Low Zone')
        self.rolls_ax.axhline(y=75, color='green', linestyle=':', alpha=0.5, label='High Zone')
        
        # Formatting
        self.rolls_ax.set_facecolor('#1E1E1E')
        self.rolls_ax.set_title('Live Roll Results', color='white', fontsize=14, weight='bold')
        self.rolls_ax.set_xlabel('Roll Number', color='white')
        self.rolls_ax.set_ylabel('Result Value', color='white')
        self.rolls_ax.tick_params(colors='white')
        self.rolls_ax.grid(True, alpha=0.3, color='gray')
        self.rolls_ax.legend(facecolor='#2B2B2B', edgecolor='white', labelcolor='white')
        
        self.rolls_canvas.draw()
        
        # Update stats
        if len(self.live_rolls) > 1:
            avg = np.mean(y)
            std = np.std(y)
            entropy = shannon_entropy(y) if len(y) > 5 else 0
            
            self.avg_label.configure(text=f"Avg: {avg:.1f}")
            self.std_label.configure(text=f"Std: {std:.1f}")
            self.entropy_label.configure(text=f"Entropy: {entropy:.2f}")
            
            # Update streak
            if len(y) >= 3:
                streaks = detect_streaks(y)
                current_streak = streaks[-1] if streaks else 0
                self.streak_label.configure(text=f"Streak: {current_streak}")
    
    def update_predictions_graph(self):
        """Update predictions graph"""
        # Placeholder for prediction accuracy graph
        pass
    
    def update_pnl_graph(self):
        """Update P&L history graph"""
        if not self.pnl_history:
            return
        
        self.pnl_ax.clear()
        
        x = list(range(len(self.pnl_history)))
        y = list(self.pnl_history)
        
        # Plot line with color based on profit/loss
        colors = ['#10B981' if val >= 0 else '#EF4444' for val in y]
        
        self.pnl_ax.plot(x, y, linewidth=3)
        
        # Fill area
        self.pnl_ax.fill_between(x, y, 0, alpha=0.3, color='#10B981' if y[-1] >= 0 else '#EF4444')
        
        # Zero line
        self.pnl_ax.axhline(y=0, color='white', linestyle='-', alpha=0.8)
        
        # Formatting
        self.pnl_ax.set_facecolor('#1E1E1E')
        self.pnl_ax.set_title('Profit & Loss History', color='white', fontsize=14, weight='bold')
        self.pnl_ax.set_xlabel('Bet Number', color='white')
        self.pnl_ax.set_ylabel('Cumulative P&L ($)', color='white')
        self.pnl_ax.tick_params(colors='white')
        self.pnl_ax.grid(True, alpha=0.3, color='gray')
        
        self.pnl_canvas.draw()
    
    def update_threshold(self, value):
        """Update confidence threshold"""
        self.threshold_label.configure(text=f"{int(value)}%")
    
    def start_update_loop(self):
        """Start periodic update loop"""
        def update():
            try:
                # Update any real-time data here
                self.update_displays()
            except:
                pass
            
            # Schedule next update
            self.root.after(5000, update)  # Update every 5 seconds
        
        update()
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()

def launch_advanced_gui():
    """Launch the Advanced Oracle GUI"""
    try:
        gui = AdvancedOracleGUI()
        gui.run()
    except Exception as e:
        print(f"❌ Failed to launch Advanced GUI: {e}")
        messagebox.showerror("Launch Error", f"Failed to launch Advanced Oracle GUI:\n{str(e)}")

if __name__ == "__main__":
    launch_advanced_gui()