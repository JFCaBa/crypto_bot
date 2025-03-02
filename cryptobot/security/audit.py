"""
Audit System
===========
Provides audit trail functionality for trading activities and system operations.
Tracks all critical actions for security and compliance purposes.
"""

import os
import json
import uuid
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

import pandas as pd
from loguru import logger


class AuditTrail:
    """
    AuditTrail class for tracking all system activities.
    """
    
    def __init__(
        self,
        audit_dir: str = "audit_logs",
        db_manager = None,
        use_db: bool = False
    ):
        """
        Initialize the audit trail.
        
        Args:
            audit_dir: Directory to store audit logs
            db_manager: Database manager instance
            use_db: Whether to use database for audit storage
        """
        self.audit_dir = audit_dir
        self.db_manager = db_manager
        self.use_db = use_db and db_manager is not None
        
        # Create audit directory if it doesn't exist
        if not os.path.exists(audit_dir):
            os.makedirs(audit_dir)
            
        # Initialize audit log file
        self.current_log_file = os.path.join(
            self.audit_dir, 
            f"audit_{datetime.now().strftime('%Y%m%d')}.log"
        )
        
        logger.info(f"Audit trail initialized, logs stored in {self.audit_dir}")
        
    def log_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        user_id: str = "system",
        severity: str = "info"
    ) -> str:
        """
        Log an event to the audit trail.
        
        Args:
            event_type: Type of event (e.g., 'trade', 'login', 'config_change')
            event_data: Event details
            user_id: User or system component that initiated the event
            severity: Event severity ('info', 'warning', 'error', 'critical')
            
        Returns:
            str: Event ID
        """
        try:
            # Generate event ID
            event_id = str(uuid.uuid4())
            
            # Create timestamp
            timestamp = datetime.now().isoformat()
            
            # Create event record
            event_record = {
                "id": event_id,
                "timestamp": timestamp,
                "type": event_type,
                "user_id": user_id,
                "severity": severity,
                "data": event_data,
                "hash": None  # Will be filled in later
            }
            
            # Calculate hash of the event (for integrity verification)
            event_record["hash"] = self._calculate_event_hash(event_record)
            
            # Store the event
            if self.use_db:
                self._store_event_db(event_record)
            else:
                self._store_event_file(event_record)
                
            logger.debug(f"Audit event logged: {event_type} - {event_id}")
            return event_id
            
        except Exception as e:
            logger.error(f"Error logging audit event: {str(e)}")
            return ""
            
    def log_trade(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        order_type: str,
        exchange: str,
        strategy: str,
        user_id: str = "system"
    ) -> str:
        """
        Log a trade event.
        
        Args:
            trade_id: Trade identifier
            symbol: Trading pair symbol
            side: Order side ('buy' or 'sell')
            amount: Order amount
            price: Order price
            order_type: Order type ('market', 'limit', etc.)
            exchange: Exchange name
            strategy: Strategy name
            user_id: User or system component that initiated the trade
            
        Returns:
            str: Event ID
        """
        trade_data = {
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "price": price,
            "order_type": order_type,
            "exchange": exchange,
            "strategy": strategy,
            "value": amount * price
        }
        
        return self.log_event("trade", trade_data, user_id)
        
    def log_login(
        self,
        user_id: str,
        ip_address: str,
        device_info: str,
        success: bool,
        failure_reason: Optional[str] = None
    ) -> str:
        """
        Log a login event.
        
        Args:
            user_id: User identifier
            ip_address: IP address
            device_info: Device information
            success: Whether login was successful
            failure_reason: Reason for failure if unsuccessful
            
        Returns:
            str: Event ID
        """
        login_data = {
            "ip_address": ip_address,
            "device_info": device_info,
            "success": success,
            "failure_reason": failure_reason
        }
        
        severity = "info" if success else "warning"
        return self.log_event("login", login_data, user_id, severity)
        
    def log_config_change(
        self,
        user_id: str,
        component: str,
        old_value: Any,
        new_value: Any,
        description: str
    ) -> str:
        """
        Log a configuration change event.
        
        Args:
            user_id: User identifier
            component: Component that was changed
            old_value: Previous value
            new_value: New value
            description: Change description
            
        Returns:
            str: Event ID
        """
        # Filter out sensitive information
        if "api_key" in str(old_value).lower() or "secret" in str(old_value).lower():
            old_value = "*** REDACTED ***"
        if "api_key" in str(new_value).lower() or "secret" in str(new_value).lower():
            new_value = "*** REDACTED ***"
            
        config_data = {
            "component": component,
            "old_value": old_value,
            "new_value": new_value,
            "description": description
        }
        
        return self.log_event("config_change", config_data, user_id)
        
    def log_error(
        self,
        error_message: str,
        error_type: str,
        stack_trace: Optional[str] = None,
        component: Optional[str] = None,
        user_id: str = "system"
    ) -> str:
        """
        Log an error event.
        
        Args:
            error_message: Error message
            error_type: Error type
            stack_trace: Stack trace
            component: Component where error occurred
            user_id: User or system component that encountered the error
            
        Returns:
            str: Event ID
        """
        error_data = {
            "error_message": error_message,
            "error_type": error_type,
            "stack_trace": stack_trace,
            "component": component
        }
        
        return self.log_event("error", error_data, user_id, "error")
        
    def log_api_request(
        self,
        exchange: str,
        endpoint: str,
        method: str,
        params: Dict[str, Any],
        status_code: Optional[int] = None,
        response_time: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> str:
        """
        Log an API request event.
        
        Args:
            exchange: Exchange name
            endpoint: API endpoint
            method: HTTP method
            params: Request parameters
            status_code: Response status code
            response_time: Response time in milliseconds
            success: Whether request was successful
            error_message: Error message if unsuccessful
            
        Returns:
            str: Event ID
        """
        # Remove sensitive information from parameters
        cleaned_params = self._clean_sensitive_data(params)
        
        api_data = {
            "exchange": exchange,
            "endpoint": endpoint,
            "method": method,
            "params": cleaned_params,
            "status_code": status_code,
            "response_time": response_time,
            "success": success,
            "error_message": error_message
        }
        
        severity = "info" if success else "warning"
        return self.log_event("api_request", api_data, "system", severity)
        
    def log_strategy_signal(
        self,
        strategy: str,
        symbol: str,
        timeframe: str,
        action: str,
        signal_data: Dict[str, Any]
    ) -> str:
        """
        Log a strategy signal event.
        
        Args:
            strategy: Strategy name
            symbol: Trading pair symbol
            timeframe: Timeframe
            action: Signal action
            signal_data: Signal details
            
        Returns:
            str: Event ID
        """
        signal_event_data = {
            "strategy": strategy,
            "symbol": symbol,
            "timeframe": timeframe,
            "action": action,
            "data": signal_data
        }
        
        return self.log_event("strategy_signal", signal_event_data)
        
    def query_events(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query audit events.
        
        Args:
            event_type: Filter by event type
            start_time: Filter by start time
            end_time: Filter by end time
            user_id: Filter by user ID
            limit: Maximum number of events to return
            
        Returns:
            list: List of event records
        """
        try:
            if self.use_db:
                return self._query_events_db(event_type, start_time, end_time, user_id, limit)
            else:
                return self._query_events_file(event_type, start_time, end_time, user_id, limit)
                
        except Exception as e:
            logger.error(f"Error querying audit events: {str(e)}")
            return []
            
    def export_events(
        self,
        output_file: str,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Export audit events to file.
        
        Args:
            output_file: Output file path
            event_type: Filter by event type
            start_time: Filter by start time
            end_time: Filter by end time
            user_id: Filter by user ID
            
        Returns:
            bool: True if export successful, False otherwise
        """
        try:
            # Query events
            events = self.query_events(event_type, start_time, end_time, user_id, limit=10000)
            
            if not events:
                logger.warning("No events to export")
                return False
                
            # Convert to DataFrame
            df = pd.DataFrame(events)
            
            # Export based on file extension
            file_ext = os.path.splitext(output_file)[1].lower()
            
            if file_ext == '.csv':
                df.to_csv(output_file, index=False)
            elif file_ext == '.json':
                df.to_json(output_file, orient='records', indent=2)
            elif file_ext == '.xlsx':
                df.to_excel(output_file, index=False)
            else:
                # Default to JSON
                output_file = f"{os.path.splitext(output_file)[0]}.json"
                df.to_json(output_file, orient='records', indent=2)
                
            logger.info(f"Exported {len(events)} audit events to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting audit events: {str(e)}")
            return False
            
    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify the integrity of audit logs by checking event hashes.
        
        Returns:
            dict: Verification results
        """
        try:
            results = {
                "verified": 0,
                "corrupted": 0,
                "corrupted_events": []
            }
            
            # Get all events
            events = self.query_events(limit=100000)
            
            for event in events:
                # Store original hash
                original_hash = event["hash"]
                
                # Calculate hash without the hash field
                event_copy = event.copy()
                event_copy["hash"] = None
                calculated_hash = self._calculate_event_hash(event_copy)
                
                # Compare hashes
                if original_hash == calculated_hash:
                    results["verified"] += 1
                else:
                    results["corrupted"] += 1
                    results["corrupted_events"].append({
                        "id": event["id"],
                        "timestamp": event["timestamp"],
                        "type": event["type"]
                    })
                    
            logger.info(
                f"Audit integrity verification: {results['verified']} verified, "
                f"{results['corrupted']} corrupted"
            )
            return results
            
        except Exception as e:
            logger.error(f"Error verifying audit integrity: {str(e)}")
            return {
                "verified": 0,
                "corrupted": 0,
                "corrupted_events": [],
                "error": str(e)
            }
            
    def _store_event_file(self, event_record: Dict[str, Any]) -> bool:
        """
        Store event to log file.
        
        Args:
            event_record: Event record
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if we need to create a new log file for the day
            log_date = datetime.now().strftime('%Y%m%d')
            expected_log_file = os.path.join(self.audit_dir, f"audit_{log_date}.log")
            
            if self.current_log_file != expected_log_file:
                self.current_log_file = expected_log_file
                
            # Append to log file
            with open(self.current_log_file, 'a') as f:
                f.write(json.dumps(event_record) + '\n')
                
            return True
            
        except Exception as e:
            logger.error(f"Error storing audit event to file: {str(e)}")
            return False
            
    def _store_event_db(self, event_record: Dict[str, Any]) -> bool:
        """
        Store event to database.
        
        Args:
            event_record: Event record
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.db_manager:
                return False
                
            # Prepare query
            query = """
            INSERT INTO audit_events (
                id, timestamp, type, user_id, severity, data, hash
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """
            
            # Execute query
            self.db_manager.execute_query(
                query,
                (
                    event_record["id"],
                    event_record["timestamp"],
                    event_record["type"],
                    event_record["user_id"],
                    event_record["severity"],
                    json.dumps(event_record["data"]),
                    event_record["hash"]
                ),
                fetch=False
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing audit event to database: {str(e)}")
            return False
            
    def _query_events_file(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query events from log files.
        
        Args:
            event_type: Filter by event type
            start_time: Filter by start time
            end_time: Filter by end time
            user_id: Filter by user ID
            limit: Maximum number of events to return
            
        Returns:
            list: List of event records
        """
        events = []
        
        # Convert dates to strings for comparison
        start_time_str = start_time.isoformat() if start_time else None
        end_time_str = end_time.isoformat() if end_time else None
        
        # Get log files
        log_files = [
            os.path.join(self.audit_dir, f)
            for f in os.listdir(self.audit_dir)
            if f.startswith("audit_") and f.endswith(".log")
        ]
        
        # Sort by date (newest first)
        log_files.sort(reverse=True)
        
        # Read events from files
        for log_file in log_files:
            if len(events) >= limit:
                break
                
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        if len(events) >= limit:
                            break
                            
                        event = json.loads(line.strip())
                        
                        # Apply filters
                        if event_type and event["type"] != event_type:
                            continue
                        if user_id and event["user_id"] != user_id:
                            continue
                        if start_time_str and event["timestamp"] < start_time_str:
                            continue
                        if end_time_str and event["timestamp"] > end_time_str:
                            continue
                            
                        events.append(event)
                        
            except Exception as e:
                logger.error(f"Error reading audit log file {log_file}: {str(e)}")
                
        return events
        
    def _query_events_db(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query events from database.
        
        Args:
            event_type: Filter by event type
            start_time: Filter by start time
            end_time: Filter by end time
            user_id: Filter by user ID
            limit: Maximum number of events to return
            
        Returns:
            list: List of event records
        """
        try:
            if not self.db_manager:
                return []
                
            # Build query with conditions
            conditions = []
            params = []
            param_index = 1
            
            if event_type:
                conditions.append(f"type = ${param_index}")
                params.append(event_type)
                param_index += 1
                
            if user_id:
                conditions.append(f"user_id = ${param_index}")
                params.append(user_id)
                param_index += 1
                
            if start_time:
                conditions.append(f"timestamp >= ${param_index}")
                params.append(start_time.isoformat())
                param_index += 1
                
            if end_time:
                conditions.append(f"timestamp <= ${param_index}")
                params.append(end_time.isoformat())
                param_index += 1
                
            # Build the WHERE clause
            where_clause = " AND ".join(conditions)
            if where_clause:
                where_clause = f"WHERE {where_clause}"
                
            # Complete query
            query = f"""
            SELECT * FROM audit_events
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ${param_index}
            """
            
            params.append(limit)
            
            # Execute query
            results = self.db_manager.execute_query(query, tuple(params))
            
            # Parse JSON data field
            for result in results:
                if "data" in result and isinstance(result["data"], str):
                    result["data"] = json.loads(result["data"])
                    
            return results
            
        except Exception as e:
            logger.error(f"Error querying audit events from database: {str(e)}")
            return []
            
    def _calculate_event_hash(self, event_record: Dict[str, Any]) -> str:
        """
        Calculate hash of event record for integrity verification.
        
        Args:
            event_record: Event record (without hash field)
            
        Returns:
            str: Hash value
        """
        # Convert to JSON string (sorted keys for consistent hashing)
        event_json = json.dumps(event_record, sort_keys=True)
        
        # Calculate SHA-256 hash
        return hashlib.sha256(event_json.encode()).hexdigest()
        
    def _clean_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove sensitive information from data.
        
        Args:
            data: Data to clean
            
        Returns:
            dict: Cleaned data
        """
        sensitive_keys = [
            "api_key", "secret", "password", "private_key", "token",
            "api_secret", "key", "credentials", "auth"
        ]
        
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    cleaned[key] = "*** REDACTED ***"
                elif isinstance(value, (dict, list)):
                    cleaned[key] = self._clean_sensitive_data(value)
                else:
                    cleaned[key] = value
            return cleaned
        elif isinstance(data, list):
            return [self._clean_sensitive_data(item) if isinstance(item, (dict, list)) else item for item in data]
        else:
            return data