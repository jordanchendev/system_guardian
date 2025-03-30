#!/usr/bin/env python3
"""
Script to start the event consumer service
This script should be run before running the message queue tests
"""
import sys
import os
import subprocess
import time
from loguru import logger

def start_consumer_service():
    """Start the event consumer service in a separate process"""
    logger.info("Starting event consumer service...")
    
    # Get the absolute path to the Python executable from the current virtual environment
    python_executable = sys.executable
    
    # Get the current working directory - this should be the project root
    current_dir = os.getcwd()
    
    # Command to start the event consumer service
    cmd = [
        python_executable,
        "-m", "system_guardian.event_consumer_service"
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            cwd=current_dir
        )
        
        # Give it a moment to start up
        time.sleep(2)
        
        # Check if the process is still running
        if process.poll() is None:
            logger.info(f"Event consumer service started successfully (PID: {process.pid})")
            logger.info("Process is running in the background. You'll need to manually terminate it when done.")
            logger.info("To test the message queue, run: python test_message_queue.py")
            return True
        else:
            stdout, stderr = process.communicate()
            logger.error(f"Event consumer service failed to start. Exit code: {process.returncode}")
            logger.error(f"STDOUT: {stdout}")
            logger.error(f"STDERR: {stderr}")
            return False
    
    except Exception as e:
        logger.error(f"Error starting event consumer service: {e}")
        return False

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    success = start_consumer_service()
    
    if success:
        logger.info("Now you can run the message queue test to verify the end-to-end flow")
        logger.info("Run: python test_message_queue.py")
    else:
        logger.error("Failed to start the event consumer service")
        sys.exit(1) 