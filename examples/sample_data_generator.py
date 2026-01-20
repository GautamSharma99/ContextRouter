"""
Sample data generator for testing the Agentic RAG system.
Creates synthetic documentation, code, and tickets for demonstration.
"""

import json
from pathlib import Path
from langchain.schema import Document

from knoroute.ingestion import (
    DocsIngestionPipeline,
    CodeIngestionPipeline,
    TicketsIngestionPipeline,
    MemoryWriter
)


def create_sample_docs():
    """Create sample documentation files."""
    docs_dir = Path("./sample_data/docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Authentication documentation
    auth_doc = """# Authentication

## Overview

Our system uses JWT (JSON Web Tokens) for authentication with the following characteristics:

- **Token Expiration**: 1 hour
- **Algorithm**: HS256
- **Storage**: httpOnly cookies

## How It Works

1. User submits credentials (username + password)
2. Server validates credentials against database
3. Server generates JWT token with user claims
4. Token is returned to client
5. Client includes token in subsequent requests

## Security Best Practices

- Always use HTTPS for token transmission
- Store tokens in httpOnly cookies to prevent XSS
- Implement token refresh mechanism
- Use strong secret keys (minimum 256 bits)

## API Endpoints

- `POST /auth/login` - Authenticate user
- `POST /auth/logout` - Invalidate token
- `POST /auth/refresh` - Refresh expired token
"""
    
    (docs_dir / "authentication.md").write_text(auth_doc)
    
    # Middleware documentation
    middleware_doc = """# Middleware

## Rate Limiting

Our API implements rate limiting using Redis:

- **Default Limit**: 100 requests per minute per IP
- **Premium Limit**: 1000 requests per minute
- **Response Header**: `X-RateLimit-Remaining`

## CORS Configuration

Cross-Origin Resource Sharing is configured to allow:

- Origins: Configurable whitelist
- Methods: GET, POST, PUT, DELETE
- Headers: Authorization, Content-Type

## Error Handling

All errors follow this format:

```json
{
  "error": "Error message",
  "code": "ERROR_CODE",
  "timestamp": "2024-01-01T00:00:00Z"
}
```
"""
    
    (docs_dir / "middleware.md").write_text(middleware_doc)
    
    print(f"✓ Created sample documentation in {docs_dir}")


def create_sample_code():
    """Create sample code files."""
    code_dir = Path("./sample_data/code")
    code_dir.mkdir(parents=True, exist_ok=True)
    
    # Authentication service
    auth_service = '''"""Authentication service implementation."""

import jwt
from datetime import datetime, timedelta
from werkzeug.security import check_password_hash

SECRET_KEY = "your-secret-key-here"  # Load from environment


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


def authenticate(username: str, password: str) -> str:
    """
    Authenticate user and return JWT token.
    
    Args:
        username: User's username
        password: User's password
        
    Returns:
        JWT token string
        
    Raises:
        AuthenticationError: If credentials are invalid
    """
    # Query user from database
    user = User.query.filter_by(username=username).first()
    
    if not user or not check_password_hash(user.password_hash, password):
        raise AuthenticationError("Invalid username or password")
    
    # Generate JWT token
    payload = {
        "user_id": user.id,
        "username": user.username,
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    
    return token


def verify_token(token: str) -> dict:
    """
    Verify JWT token and return payload.
    
    Args:
        token: JWT token string
        
    Returns:
        Token payload dictionary
        
    Raises:
        AuthenticationError: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError:
        raise AuthenticationError("Invalid token")
'''
    
    (code_dir / "auth_service.py").write_text(auth_service)
    
    # Rate limiting middleware
    rate_limit = '''"""Rate limiting middleware using Redis."""

import redis
from datetime import datetime
from flask import request, jsonify

redis_client = redis.Redis(host='localhost', port=6379, db=0)

RATE_LIMIT = 100  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds


def rate_limit_middleware():
    """
    Rate limiting middleware.
    
    Limits requests per IP address using sliding window algorithm.
    """
    # Get client IP
    client_ip = request.remote_addr
    
    # Create Redis key
    key = f"rate_limit:{client_ip}"
    
    # Get current request count
    current_count = redis_client.get(key)
    
    if current_count is None:
        # First request in window
        redis_client.setex(key, RATE_LIMIT_WINDOW, 1)
        remaining = RATE_LIMIT - 1
    else:
        current_count = int(current_count)
        
        if current_count >= RATE_LIMIT:
            # Rate limit exceeded
            return jsonify({
                "error": "Rate limit exceeded",
                "code": "RATE_LIMIT_EXCEEDED"
            }), 429
        
        # Increment counter
        redis_client.incr(key)
        remaining = RATE_LIMIT - current_count - 1
    
    # Add rate limit headers
    response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    
    return None
'''
    
    (code_dir / "rate_limit.py").write_text(rate_limit)
    
    print(f"✓ Created sample code in {code_dir}")


def create_sample_tickets():
    """Create sample ticket data."""
    tickets_dir = Path("./sample_data/tickets")
    tickets_dir.mkdir(parents=True, exist_ok=True)
    
    tickets = [
        {
            "id": "TICKET-001",
            "title": "Authentication fails with expired tokens",
            "description": "Users are experiencing authentication failures when their JWT tokens expire. The system should automatically refresh tokens before expiration.",
            "status": "closed",
            "severity": "high",
            "created_at": "2024-01-15T10:30:00Z",
            "comments": [
                "Root cause: Token refresh endpoint was not implemented",
                "Fixed by adding /auth/refresh endpoint",
                "Deployed to production on 2024-01-16"
            ]
        },
        {
            "id": "TICKET-002",
            "title": "Rate limiting not working for authenticated users",
            "description": "Rate limiting middleware is applying limits to all users including authenticated premium users who should have higher limits.",
            "status": "closed",
            "severity": "medium",
            "created_at": "2024-01-20T14:15:00Z",
            "comments": [
                "Issue: Middleware checks IP before checking auth status",
                "Solution: Check user tier from JWT token first",
                "Updated rate_limit.py to handle premium users"
            ]
        },
        {
            "id": "TICKET-003",
            "title": "CORS errors on production",
            "description": "Frontend application is getting CORS errors when calling API endpoints from production domain.",
            "status": "resolved",
            "severity": "critical",
            "created_at": "2024-01-22T09:00:00Z",
            "comments": [
                "Production domain was not in CORS whitelist",
                "Added production domain to allowed origins",
                "Verified fix in production"
            ]
        }
    ]
    
    with open(tickets_dir / "tickets.json", "w") as f:
        json.dump(tickets, f, indent=2)
    
    print(f"✓ Created sample tickets in {tickets_dir}")


def ingest_all_sample_data():
    """Ingest all sample data into vector stores."""
    print("\n" + "="*60)
    print("Ingesting Sample Data")
    print("="*60 + "\n")
    
    # Ingest documentation
    print("1. Ingesting documentation...")
    docs_pipeline = DocsIngestionPipeline()
    docs_pipeline.ingest_directory(
        directory_path="./sample_data/docs",
        glob_pattern="**/*.md",
        doc_type="guide"
    )
    
    # Ingest code
    print("\n2. Ingesting code...")
    code_pipeline = CodeIngestionPipeline()
    code_pipeline.ingest_directory(
        directory_path="./sample_data/code",
        glob_pattern="**/*.py",
        extract_functions=True
    )
    
    # Ingest tickets
    print("\n3. Ingesting tickets...")
    tickets_pipeline = TicketsIngestionPipeline()
    tickets_pipeline.ingest_json(
        file_path="./sample_data/tickets/tickets.json",
        format_type="generic"
    )
    
    # Add some initial memory
    print("\n4. Adding initial memory...")
    memory_writer = MemoryWriter()
    
    insights = [
        {
            "insight": "JWT tokens should be refreshed before expiration to prevent authentication failures",
            "learned_from": "Historical ticket analysis",
            "confidence": 0.95,
            "tags": ["authentication", "best-practice"]
        },
        {
            "insight": "Rate limiting should consider user tier from JWT claims, not just IP address",
            "learned_from": "Rate limiting bug fix",
            "confidence": 0.9,
            "tags": ["rate-limiting", "authentication"]
        },
        {
            "insight": "Always include production domains in CORS whitelist before deployment",
            "learned_from": "CORS production incident",
            "confidence": 0.95,
            "tags": ["cors", "deployment"]
        }
    ]
    
    memory_writer.write_multiple_insights(insights)
    
    print("\n" + "="*60)
    print("✓ All sample data ingested successfully!")
    print("="*60)


if __name__ == "__main__":
    print("Creating sample data for Agentic RAG system...\n")
    
    # Create sample data
    create_sample_docs()
    create_sample_code()
    create_sample_tickets()
    
    print("\n" + "="*60)
    print("Sample data created successfully!")
    print("="*60)
    
    # Ask if user wants to ingest
    response = input("\nDo you want to ingest this data into vector stores? (y/n): ")
    
    if response.lower() == 'y':
        ingest_all_sample_data()
    else:
        print("\nSkipping ingestion. Run this script again to ingest later.")
