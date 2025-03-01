"""
Web Dashboard
============
Web-based dashboard for monitoring and controlling the trading bot.
"""

import os
import json
import datetime
from typing import Dict, List, Any

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
from flask import Flask

from loguru import logger
from cryptobot.core.engine import TradingEngine


class WebDashboard:
    """Web dashboard for monitoring and controlling the trading bot."""
    
    def __init__(
        self,
        trading_engine: TradingEngine,
        port: int = 8050,
        debug: bool = False,
        theme: str = 'darkly'
    ):
        """
        Initialize the web dashboard.
        
        Args:
            trading_engine: TradingEngine instance
            port: Port to run the dashboard on
            debug: Whether to run in debug mode
            theme: Bootstrap theme
        """
        self.trading_engine = trading_engine
        self.port = port
        self.debug = debug
        
        # Create Flask server
        self.server = Flask(__name__)
        
        # Create Dash app
        self.app = dash.Dash(
            __name__,
            server=self.server,
            external_stylesheets=[dbc.themes.DARKLY if theme == 'darkly' else dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        
        # Set app title
        self.app.title = "CryptoBot Dashboard"
        
        # Initialize the layout
        self._setup_layout()
        
        # Setup callbacks
        self._setup_callbacks()
        
        logger.info(f"Web dashboard initialized, will run on port {port}")
        
    def run(self):
        """Run the dashboard server."""
        self.app.run_server(debug=self.debug, port=self.port)
        
    def _setup_layout(self):
        """Setup the dashboard layout."""
        # Navbar
        navbar = dbc.Navbar(
            dbc.Container(
                [
                    dbc.Row([
                        dbc.Col(html.Img(src="/assets/logo.png", height="30px"), width="auto"),
                        dbc.Col(dbc.NavbarBrand("CryptoBot Dashboard", className="ms-2"), width="auto"),
                    ]),
                    dbc.NavbarToggler(id="navbar-toggler"),
                    dbc.Collapse(
                        dbc.Nav([
                            dbc.NavItem(dbc.NavLink("Overview", href="#")),
                            dbc.NavItem(dbc.NavLink("Strategies", href="#")),
                            dbc.NavItem(dbc.NavLink("Trades", href="#")),
                            dbc.NavItem(dbc.NavLink("Settings", href="#")),
                        ], className="ms-auto", navbar=True),
                        id="navbar-collapse",
                        navbar=True,
                    ),
                ],
                fluid=True,
            ),
            color="dark",
            dark=True,
            className="mb-4",
        )
        
        # Bot status card
        bot_status_card = dbc.Card(
            dbc.CardBody([
                html.H4("Bot Status", className="card-title"),
                html.Div(id="bot-status-content", children=[
                    dbc.Badge("Offline", color="danger", className="me-1"),
                    html.Span("Last update: Never", id="last-update-time"),
                    html.Div([
                        dbc.Button("Start", color="success", id="start-bot-button", className="me-2"),
                        dbc.Button("Stop", color="danger", id="stop-bot-button", disabled=True),
                    ], className="mt-2")
                ])
            ]),
            className="mb-4"
        )
        
        # Account balance card
        account_balance_card = dbc.Card(
            dbc.CardBody([
                html.H4("Account Balance", className="card-title"),
                html.Div(id="account-balance-content", children=[
                    html.H2("$0.00", id="total-balance"),
                    dcc.Graph(id="balance-history-graph", config={'displayModeBar': False})
                ])
            ]),
            className="mb-4"
        )
        
        # Active positions card
        active_positions_card = dbc.Card(
            dbc.CardBody([
                html.H4("Active Positions", className="card-title"),
                html.Div(id="active-positions-content", children=[
                    html.P("No active positions", id="no-positions-msg"),
                    html.Div(id="positions-table-container", style={"display": "none"}, children=[
                        dbc.Table(id="positions-table", bordered=True, striped=True, hover=True)
                    ])
                ])
            ]),
            className="mb-4"
        )
        
        # Trading history card
        trading_history_card = dbc.Card(
            dbc.CardBody([
                html.H4("Trading History", className="card-title"),
                html.Div(id="trading-history-content", children=[
                    dbc.Table(id="trades-table", bordered=True, striped=True, hover=True),
                ])
            ]),
            className="mb-4"
        )
        
        # Strategy performance card
        strategy_performance_card = dbc.Card(
            dbc.CardBody([
                html.H4("Strategy Performance", className="card-title"),
                html.Div(id="strategy-performance-content", children=[
                    dcc.Graph(id="strategy-performance-graph", config={'displayModeBar': False})
                ])
            ]),
            className="mb-4"
        )
        
        # Layout structure
        self.app.layout = html.Div([
            navbar,
            dbc.Container([
                # Status row
                dbc.Row([
                    dbc.Col(bot_status_card, width=4),
                    dbc.Col(account_balance_card, width=8),
                ]),
                
                # Positions and history row
                dbc.Row([
                    dbc.Col(active_positions_card, width=4),
                    dbc.Col(trading_history_card, width=8),
                ]),
                
                # Strategy row
                dbc.Row([
                    dbc.Col(strategy_performance_card),
                ]),
                
                # Update interval
                dcc.Interval(
                    id='interval-component',
                    interval=5000,  # 5 seconds in milliseconds
                    n_intervals=0
                ),
            ], fluid=True)
        ])
        
    def _setup_callbacks(self):
        """Setup the dashboard callbacks."""
        # Update bot status
        @self.app.callback(
            [Output("bot-status-content", "children"),
             Output("last-update-time", "children")],
            [Input("interval-component", "n_intervals"),
             Input("start-bot-button", "n_clicks"),
             Input("stop-bot-button", "n_clicks")],
            prevent_initial_call=False
        )
        async def update_bot_status(n_intervals, start_clicks, stop_clicks):
            # Get bot status
            status = await self.trading_engine.get_status()
            
            # Update status badge
            if status["is_running"]:
                status_badge = dbc.Badge("Online", color="success", className="me-1")
                start_button = dbc.Button("Start", color="success", id="start-bot-button", className="me-2", disabled=True)
                stop_button = dbc.Button("Stop", color="danger", id="stop-bot-button")
            else:
                status_badge = dbc.Badge("Offline", color="danger", className="me-1")
                start_button = dbc.Button("Start", color="success", id="start-bot-button", className="me-2")
                stop_button = dbc.Button("Stop", color="danger", id="stop-bot-button", disabled=True)
                
            # Format last update time
            last_update = f"Last update: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Return updated content
            return [
                [
                    status_badge,
                    html.Span(last_update, id="last-update-time"),
                    html.Div([
                        start_button,
                        stop_button,
                    ], className="mt-2")
                ],
                last_update
            ]
        
        # Update account balance
        @self.app.callback(
            [Output("total-balance", "children"),
             Output("balance-history-graph", "figure")],
            [Input("interval-component", "n_intervals")],
            prevent_initial_call=False
        )
        async def update_account_balance(n_intervals):
            # Get status including account balance
            status = await self.trading_engine.get_status()
            
            # Extract balance info
            total_balance = 0
            for exchange_id, balance_info in status.get("account_balances", {}).items():
                # In a real implementation, we would extract the total balance
                # For now, we'll use a placeholder
                if "total" in balance_info:
                    total_balance += balance_info["total"].get("USDT", 0)
                elif "info" in balance_info and "balances" in balance_info["info"]:
                    for asset in balance_info["info"]["balances"]:
                        if asset["asset"] in ["USDT", "BUSD", "USD", "USDC"]:
                            total_balance += float(asset["free"]) + float(asset["locked"])
            
            # Format balance display
            balance_display = f"${total_balance:.2f}"
            
            # Create a mock balance history figure
            # In a real implementation, this would use actual historical data
            df = pd.DataFrame({
                "date": pd.date_range(start="2023-01-01", periods=30, freq="D"),
                "balance": [total_balance * (1 + i * 0.01) for i in range(30)]
            })
            
            fig = px.line(
                df, 
                x="date", 
                y="balance",
                labels={"date": "Date", "balance": "Balance (USDT)"},
                template="plotly_dark"
            )
            
            fig.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=200,
                hovermode="x unified"
            )
            
            return balance_display, fig
        
        # Update active positions
        @self.app.callback(
            [Output("no-positions-msg", "style"),
             Output("positions-table-container", "style"),
             Output("positions-table", "children")],
            [Input("interval-component", "n_intervals")],
            prevent_initial_call=False
        )
        async def update_active_positions(n_intervals):
            # Get status including active positions
            status = await self.trading_engine.get_status()
            
            # Check if there are active positions
            has_positions = False
            positions_data = []
            
            for strategy_id, strategy_info in status.get("strategies", {}).items():
                for symbol, position in strategy_info.get("positions", {}).items():
                    if position.get("is_active", False):
                        has_positions = True
                        positions_data.append({
                            "symbol": symbol,
                            "side": position.get("side", ""),
                            "entry_price": position.get("entry_price", 0),
                            "amount": position.get("amount", 0),
                            "entry_time": position.get("entry_time", ""),
                            "strategy": strategy_id
                        })
            
            # Display logic
            if has_positions:
                no_positions_style = {"display": "none"}
                table_container_style = {"display": "block"}
                
                # Create table header
                table_header = html.Thead(html.Tr([
                    html.Th("Symbol"),
                    html.Th("Side"),
                    html.Th("Entry Price"),
                    html.Th("Amount"),
                    html.Th("Entry Time"),
                    html.Th("Strategy")
                ]))
                
                # Create table rows
                table_rows = [
                    html.Tr([
                        html.Td(pos["symbol"]),
                        html.Td(pos["side"].title()),
                        html.Td(f"{pos['entry_price']:.8f}"),
                        html.Td(f"{pos['amount']:.8f}"),
                        html.Td(pos["entry_time"]),
                        html.Td(pos["strategy"])
                    ])
                    for pos in positions_data
                ]
                
                table_body = html.Tbody(table_rows)
                
                return no_positions_style, table_container_style, [table_header, table_body]
            else:
                return {"display": "block"}, {"display": "none"}, []
        
        # Update trading history
        @self.app.callback(
            Output("trades-table", "children"),
            [Input("interval-component", "n_intervals")],
            prevent_initial_call=False
        )
        async def update_trading_history(n_intervals):
            # Get status including trading history
            status = await self.trading_engine.get_status()
            
            # Collect trade data from all strategies
            all_trades = []
            for strategy_id, strategy_info in status.get("strategies", {}).items():
                # In a real implementation, we would get the actual trade history
                # For now, we'll use placeholder data
                trades = strategy_info.get("performance", {}).get("trades", [])
                if not trades:
                    continue
                
                for trade in trades:
                    trade["strategy"] = strategy_id
                    all_trades.append(trade)
            
            # Sort by timestamp (most recent first)
            all_trades = sorted(all_trades, key=lambda x: x.get("timestamp", ""), reverse=True)
            
            # Limit to the most recent 10 trades
            all_trades = all_trades[:10]
            
            # Create table header
            table_header = html.Thead(html.Tr([
                html.Th("Time"),
                html.Th("Symbol"),
                html.Th("Side"),
                html.Th("Price"),
                html.Th("Amount"),
                html.Th("Strategy"),
                html.Th("PnL")
            ]))
            
            # Create table rows
            if all_trades:
                table_rows = [
                    html.Tr([
                        html.Td(trade.get("timestamp", "")),
                        html.Td(trade.get("symbol", "")),
                        html.Td(trade.get("side", "").title()),
                        html.Td(f"{trade.get('price', 0):.8f}"),
                        html.Td(f"{trade.get('amount', 0):.8f}"),
                        html.Td(trade.get("strategy", "")),
                        html.Td([
                            html.Span(
                                f"{trade.get('pnl', 0):.2f} ({trade.get('pnl_percent', 0):.2f}%)",
                                style={"color": "green" if trade.get("pnl", 0) >= 0 else "red"}
                            )
                        ])
                    ])
                    for trade in all_trades
                ]
            else:
                # If no trades, show a placeholder row
                table_rows = [html.Tr([html.Td(colspan=7, style={"text-align": "center"}, children="No trading history yet")])]
            
            table_body = html.Tbody(table_rows)
            
            return [table_header, table_body]
        
        # Update strategy performance
        @self.app.callback(
            Output("strategy-performance-graph", "figure"),
            [Input("interval-component", "n_intervals")],
            prevent_initial_call=False
        )
        async def update_strategy_performance(n_intervals):
            # Get status including strategy performance
            status = await self.trading_engine.get_status()
            
            # Collect performance data for all strategies
            performance_data = []
            for strategy_id, strategy_info in status.get("strategies", {}).items():
                perf = strategy_info.get("performance", {})
                performance_data.append({
                    "strategy": strategy_id,
                    "win_rate": perf.get("win_rate", 0),
                    "total_pnl_percent": perf.get("total_pnl_percent", 0),
                    "profit_factor": perf.get("profit_factor", 0),
                    "total_trades": perf.get("total_trades", 0)
                })
            
            # Create a performance comparison figure
            if performance_data:
                # Convert to DataFrame
                df = pd.DataFrame(performance_data)
                
                # Create a bar chart
                fig = go.Figure()
                
                # Add win rate bars
                fig.add_trace(go.Bar(
                    x=df["strategy"],
                    y=df["win_rate"],
                    name="Win Rate (%)",
                    marker_color="green"
                ))
                
                # Add PnL bars
                fig.add_trace(go.Bar(
                    x=df["strategy"],
                    y=df["total_pnl_percent"],
                    name="Total PnL (%)",
                    marker_color="blue"
                ))
                
                # Add profit factor as a line
                fig.add_trace(go.Scatter(
                    x=df["strategy"],
                    y=df["profit_factor"],
                    name="Profit Factor",
                    mode="markers+lines",
                    marker=dict(size=10),
                    line=dict(color="orange", width=2),
                    yaxis="y2"
                ))
                
                # Update layout
                fig.update_layout(
                    title="Strategy Performance Comparison",
                    barmode="group",
                    yaxis=dict(
                        title="Percentage (%)",
                        titlefont=dict(color="green"),
                        tickfont=dict(color="green")
                    ),
                    yaxis2=dict(
                        title="Profit Factor",
                        titlefont=dict(color="orange"),
                        tickfont=dict(color="orange"),
                        anchor="x",
                        overlaying="y",
                        side="right"
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    template="plotly_dark",
                    hovermode="x unified"
                )
            else:
                # Create an empty figure with a message
                fig = go.Figure()
                fig.add_annotation(
                    text="No strategy performance data available",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=16)
                )
                
                fig.update_layout(
                    template="plotly_dark",
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
            
            return fig
        
        # Start bot button callback
        @self.app.callback(
            Output("start-bot-button", "disabled"),
            [Input("start-bot-button", "n_clicks")],
            prevent_initial_call=True
        )
        async def start_bot(n_clicks):
            if n_clicks:
                success = await self.trading_engine.start()
                return success
            return dash.no_update
        
        # Stop bot button callback
        @self.app.callback(
            Output("stop-bot-button", "disabled"),
            [Input("stop-bot-button", "n_clicks")],
            prevent_initial_call=True
        )
        async def stop_bot(n_clicks):
            if n_clicks:
                success = await self.trading_engine.stop()
                return not success  # If stop successful, disable the stop button
            return dash.no_update