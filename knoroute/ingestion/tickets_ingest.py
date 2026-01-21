"""Tickets ingestion pipeline."""

import json
import csv
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from langchain_core.documents import Document

from knoroute.vectorstores import TicketsVectorStore


class TicketsIngestionPipeline:
    """
    Ingestion pipeline for ticket/issue data.
    Supports JSON and CSV formats from Jira, GitHub Issues, etc.
    """
    
    def __init__(self, vector_store: Optional[TicketsVectorStore] = None):
        """
        Initialize the tickets ingestion pipeline.
        
        Args:
            vector_store: TicketsVectorStore instance (creates new if None)
        """
        self.vector_store = vector_store or TicketsVectorStore()
    
    def load_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load tickets from a JSON file.
        
        Expected format:
        [
            {
                "id": "TICKET-123",
                "title": "Bug title",
                "description": "Bug description",
                "status": "closed",
                "severity": "high",
                "created_at": "2024-01-01T00:00:00Z",
                "comments": ["comment1", "comment2"]
            },
            ...
        ]
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of ticket dictionaries
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both single object and array
        if isinstance(data, dict):
            data = [data]
        
        return data
    
    def load_csv_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load tickets from a CSV file.
        
        Expected columns: id, title, description, status, severity, created_at
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of ticket dictionaries
        """
        tickets = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tickets.append(row)
        
        return tickets
    
    def parse_github_issues(self, issues: List[Dict[str, Any]]) -> List[Document]:
        """
        Parse GitHub Issues format.
        
        Args:
            issues: List of GitHub issue dictionaries
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for issue in issues:
            # Combine title, body, and comments
            content_parts = [
                f"Title: {issue.get('title', '')}",
                f"\nDescription: {issue.get('body', '')}",
            ]
            
            # Add comments if available
            if 'comments' in issue and issue['comments']:
                content_parts.append("\nComments:")
                if isinstance(issue['comments'], list):
                    for comment in issue['comments']:
                        content_parts.append(f"- {comment}")
                else:
                    content_parts.append(f"- {issue['comments']}")
            
            content = "\n".join(content_parts)
            
            # Extract metadata
            metadata = {
                "ticket_id": str(issue.get('number', issue.get('id', ''))),
                "status": issue.get('state', 'open'),
                "severity": self._infer_severity(issue),
                "created_at": issue.get('created_at', datetime.now().isoformat()),
            }
            
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        
        return documents
    
    def parse_jira_issues(self, issues: List[Dict[str, Any]]) -> List[Document]:
        """
        Parse Jira Issues format.
        
        Args:
            issues: List of Jira issue dictionaries
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for issue in issues:
            # Handle Jira's nested structure
            fields = issue.get('fields', issue)
            
            # Combine summary and description
            content_parts = [
                f"Title: {fields.get('summary', '')}",
                f"\nDescription: {fields.get('description', '')}",
            ]
            
            # Add comments if available
            if 'comment' in fields and 'comments' in fields['comment']:
                content_parts.append("\nComments:")
                for comment in fields['comment']['comments']:
                    content_parts.append(f"- {comment.get('body', '')}")
            
            content = "\n".join(content_parts)
            
            # Extract metadata
            metadata = {
                "ticket_id": issue.get('key', issue.get('id', '')),
                "status": fields.get('status', {}).get('name', 'open'),
                "severity": fields.get('priority', {}).get('name', 'medium').lower(),
                "created_at": fields.get('created', datetime.now().isoformat()),
            }
            
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        
        return documents
    
    def parse_generic_tickets(self, tickets: List[Dict[str, Any]]) -> List[Document]:
        """
        Parse generic ticket format.
        
        Args:
            tickets: List of ticket dictionaries
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for ticket in tickets:
            # Combine available fields
            content_parts = []
            
            if 'title' in ticket:
                content_parts.append(f"Title: {ticket['title']}")
            
            if 'description' in ticket:
                content_parts.append(f"\nDescription: {ticket['description']}")
            
            if 'comments' in ticket:
                content_parts.append("\nComments:")
                comments = ticket['comments']
                if isinstance(comments, list):
                    for comment in comments:
                        content_parts.append(f"- {comment}")
                else:
                    content_parts.append(f"- {comments}")
            
            content = "\n".join(content_parts)
            
            # Extract metadata
            metadata = {
                "ticket_id": str(ticket.get('id', ticket.get('ticket_id', ''))),
                "status": ticket.get('status', 'open'),
                "severity": ticket.get('severity', ticket.get('priority', 'medium')).lower(),
                "created_at": ticket.get('created_at', datetime.now().isoformat()),
            }
            
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        
        return documents
    
    def ingest_json(
        self,
        file_path: str,
        format_type: str = "generic"
    ) -> List[str]:
        """
        Ingest tickets from a JSON file.
        
        Args:
            file_path: Path to JSON file
            format_type: Format type (generic, github, jira)
            
        Returns:
            List of document IDs
        """
        # Load tickets
        tickets = self.load_json_file(file_path)
        
        # Parse based on format
        if format_type == "github":
            documents = self.parse_github_issues(tickets)
        elif format_type == "jira":
            documents = self.parse_jira_issues(tickets)
        else:
            documents = self.parse_generic_tickets(tickets)
        
        # Add to vector store
        doc_ids = self.vector_store.add_documents(documents)
        
        print(f"✓ Ingested {len(doc_ids)} tickets from {file_path}")
        return doc_ids
    
    def ingest_csv(self, file_path: str) -> List[str]:
        """
        Ingest tickets from a CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of document IDs
        """
        # Load tickets
        tickets = self.load_csv_file(file_path)
        
        # Parse as generic format
        documents = self.parse_generic_tickets(tickets)
        
        # Add to vector store
        doc_ids = self.vector_store.add_documents(documents)
        
        print(f"✓ Ingested {len(doc_ids)} tickets from {file_path}")
        return doc_ids
    
    def _infer_severity(self, issue: Dict[str, Any]) -> str:
        """
        Infer severity from issue labels or priority.
        
        Args:
            issue: Issue dictionary
            
        Returns:
            Severity level
        """
        # Check labels
        labels = issue.get('labels', [])
        if isinstance(labels, list):
            label_names = [l.get('name', '').lower() if isinstance(l, dict) else str(l).lower() for l in labels]
            
            if any('critical' in l or 'urgent' in l for l in label_names):
                return 'critical'
            elif any('high' in l for l in label_names):
                return 'high'
            elif any('low' in l for l in label_names):
                return 'low'
        
        # Check priority field
        priority = issue.get('priority', '')
        if priority:
            return priority.lower()
        
        return 'medium'


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = TicketsIngestionPipeline()
    
    # Example: Ingest GitHub issues
    # pipeline.ingest_json("github_issues.json", format_type="github")
    
    # Example: Ingest Jira issues
    # pipeline.ingest_json("jira_issues.json", format_type="jira")
    
    # Example: Ingest CSV
    # pipeline.ingest_csv("tickets.csv")
    
    print("Tickets ingestion pipeline ready.")
    print("Use pipeline.ingest_json() or pipeline.ingest_csv() to add tickets.")
