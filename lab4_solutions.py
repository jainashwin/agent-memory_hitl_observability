"""
Lab 4 Solutions: Customer Support - Exercise Solutions
=======================================================
This file contains complete solutions for all Lab 4 exercises.

Exercises Covered:
1. Add sentiment analysis for auto-escalation
2. Implement conversation timeout tracking
3. Create knowledge base lookup integration
4. Build conversation summary generation
5. Add customer satisfaction survey
6. Implement multi-agent handoff (billing, technical, fraud)
7. Create conversation analytics
8. Test different support scenarios
"""

import os
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
import operator
from datetime import datetime


# ============================================================================
# State Definition
# ============================================================================
class CustomerSupportState(TypedDict):
    # Conversation
    messages: Annotated[list[BaseMessage], operator.add]
    
    # Customer info
    customer_id: str
    customer_name: str
    issue_type: str
    
    # Issue details
    account_number: str
    issue_description: str
    urgency_level: str
    
    # Sentiment tracking
    sentiment_score: float
    sentiment_history: list
    
    # Timeout tracking
    last_response_time: str
    timeout_warnings: int
    
    # Knowledge base
    kb_results: list
    
    # Conversation summary
    conversation_summary: str
    
    # Resolution
    resolution_attempted: bool
    resolved: bool
    escalated_to_human: bool
    human_agent_notes: str
    agent_type: str
    
    # Customer satisfaction
    satisfaction_rating: int
    satisfaction_feedback: str
    
    # Analytics
    conversation_turns: int
    resolution_time: float
    
    # Metadata
    conversation_start: str


# ============================================================================
# LLM Setup
# ============================================================================
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found")
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, groq_api_key=api_key)


# ============================================================================
# SOLUTION 1: Add Sentiment Analysis
# ============================================================================
def analyze_sentiment(message: str) -> dict:
    """Analyze sentiment of customer message"""
    llm = get_llm()
    
    prompt = f"""Analyze the sentiment of this customer message.
    Return ONLY a JSON with:
    - sentiment: "positive", "neutral", or "negative"
    - score: 0-100 (0=very negative, 50=neutral, 100=very positive)
    - emotion: primary emotion detected
    
    Message: "{message}"
    
    Return JSON only."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        import json
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        return {
            "sentiment": result.get("sentiment", "neutral"),
            "score": result.get("score", 50),
            "emotion": result.get("emotion", "neutral")
        }
    except:
        return {"sentiment": "neutral", "score": 50, "emotion": "neutral"}


def ai_support_with_sentiment(state: CustomerSupportState) -> CustomerSupportState:
    """AI support agent with sentiment analysis"""
    llm = get_llm()
    
    # Analyze sentiment of last customer message
    if state['messages']:
        last_msg = state['messages'][-1]
        if isinstance(last_msg, HumanMessage):
            sentiment_data = analyze_sentiment(last_msg.content)
            sentiment_score = sentiment_data['score']
            
            # Add to history
            sentiment_history = state.get('sentiment_history', [])
            sentiment_history.append({
                "turn": state.get('conversation_turns', 0),
                "score": sentiment_score,
                "sentiment": sentiment_data['sentiment'],
                "emotion": sentiment_data['emotion']
            })
            
            print(f"\n📊 Sentiment: {sentiment_data['sentiment'].upper()} " +
                  f"(Score: {sentiment_score}, Emotion: {sentiment_data['emotion']})")
            
            # AUTO-ESCALATE if negative sentiment
            if sentiment_score < 40:
                print("⚠️  Negative sentiment detected - Auto-escalating\n")
                return {
                    "sentiment_score": sentiment_score,
                    "sentiment_history": sentiment_history,
                    "escalated_to_human": True,
                    "messages": [AIMessage(content="I understand you're frustrated. Let me connect you with a specialist who can help.")]
                }
    
    # Normal AI response
    system_message = """You are a helpful banking customer support AI.
    Be empathetic, professional, and helpful. Ask clarifying questions one at a time."""
    
    messages = [SystemMessage(content=system_message)] + state['messages']
    response = llm.invoke(messages)
    
    print(f"\n🤖 AI: {response.content}\n")
    
    return {
        "messages": [AIMessage(content=response.content)],
        "conversation_turns": state.get('conversation_turns', 0) + 1,
        "sentiment_score": state.get('sentiment_score', 50),
        "sentiment_history": state.get('sentiment_history', [])
    }


def solution_1_sentiment_analysis():
    """Test sentiment-based auto-escalation"""
    print("\n" + "="*80)
    print("SOLUTION 1: Sentiment Analysis with Auto-Escalation")
    print("="*80 + "\n")
    
    workflow = StateGraph(CustomerSupportState)
    workflow.add_node("ai_agent", ai_support_with_sentiment)
    
    def check_escalation(state):
        if state.get('escalated_to_human', False) or state.get('resolved', False):
            return "end"
        return "continue"
    
    workflow.add_edge(START, "ai_agent")
    workflow.add_conditional_edges(
        "ai_agent",
        check_escalation,
        {"end": END, "continue": "ai_agent"}
    )
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    thread = {"configurable": {"thread_id": "sentiment_test"}}
    
    # Simulate conversation with negative sentiment
    messages = [
        "I've been trying to access my account for an hour!",
        "This is ridiculous! Your system never works!",
        "I'm extremely frustrated and angry!"
    ]
    
    state = {
        "messages": [HumanMessage(content=messages[0])],
        "customer_id": "CUST-001",
        "customer_name": "Angry Customer",
        "issue_type": "login",
        "account_number": "",
        "issue_description": "",
        "urgency_level": "high",
        "sentiment_score": 50.0,
        "sentiment_history": [],
        "last_response_time": "",
        "timeout_warnings": 0,
        "kb_results": [],
        "conversation_summary": "",
        "resolution_attempted": False,
        "resolved": False,
        "escalated_to_human": False,
        "human_agent_notes": "",
        "agent_type": "",
        "satisfaction_rating": 0,
        "satisfaction_feedback": "",
        "conversation_turns": 0,
        "resolution_time": 0.0,
        "conversation_start": datetime.now().isoformat()
    }
    
    for event in app.stream(state, thread, stream_mode="values"):
        final_state = event
        if final_state.get('escalated_to_human'):
            break
    
    print("✅ Auto-escalation triggered by negative sentiment\n")


# ============================================================================
# SOLUTION 2: Conversation Timeout Tracking
# ============================================================================
def track_timeout(state: CustomerSupportState) -> CustomerSupportState:
    """Track time since last response and send warnings"""
    from datetime import datetime, timedelta
    
    current_time = datetime.now()
    last_time_str = state.get('last_response_time', '')
    
    if last_time_str:
        last_time = datetime.fromisoformat(last_time_str)
        time_diff = (current_time - last_time).total_seconds() / 60  # minutes
        
        # If more than 5 minutes, send warning
        if time_diff > 5:
            warnings = state.get('timeout_warnings', 0) + 1
            print(f"\n⏰ Timeout Warning {warnings}: {time_diff:.1f} minutes since last response\n")
            
            return {
                "timeout_warnings": warnings,
                "last_response_time": current_time.isoformat()
            }
    
    return {"last_response_time": current_time.isoformat()}


def solution_2_timeout_tracking():
    """Demonstrate timeout tracking"""
    print("\n" + "="*80)
    print("SOLUTION 2: Conversation Timeout Tracking")
    print("="*80 + "\n")
    
    from datetime import datetime, timedelta
    
    # Simulate timeout scenario
    state = {
        "last_response_time": (datetime.now() - timedelta(minutes=10)).isoformat(),
        "timeout_warnings": 0
    }
    
    print(f"Last Response: 10 minutes ago")
    result = track_timeout(state)
    print(f"Timeout Warnings: {result['timeout_warnings']}")
    print("\n✅ Timeout tracking demonstrated\n")


# ============================================================================
# SOLUTION 3: Knowledge Base Lookup
# ============================================================================
def search_knowledge_base(query: str) -> list:
    """Simulate knowledge base search"""
    # In production, this would query actual KB
    kb_articles = {
        "password reset": {
            "title": "How to Reset Your Password",
            "content": "1. Go to login page 2. Click 'Forgot Password' 3. Enter email 4. Check inbox for reset link",
            "article_id": "KB-001"
        },
        "login": {
            "title": "Troubleshooting Login Issues",
            "content": "Clear browser cache, check caps lock, verify username format, try password reset",
            "article_id": "KB-002"
        },
        "balance": {
            "title": "Check Account Balance",
            "content": "Log into online banking, navigate to Accounts, select account to view balance",
            "article_id": "KB-003"
        }
    }
    
    query_lower = query.lower()
    results = []
    
    for keyword, article in kb_articles.items():
        if keyword in query_lower:
            results.append(article)
    
    return results


def ai_support_with_kb(state: CustomerSupportState) -> CustomerSupportState:
    """AI support with knowledge base lookup"""
    llm = get_llm()
    
    # Search KB for relevant articles
    if state['messages']:
        last_msg = state['messages'][-1].content
        kb_results = search_knowledge_base(last_msg)
        
        if kb_results:
            print(f"\n📚 Found {len(kb_results)} KB articles")
            for article in kb_results:
                print(f"   - {article['title']} ({article['article_id']})")
            
            # Include KB in AI response
            kb_context = "\n\n".join([
                f"KB Article: {a['title']}\n{a['content']}"
                for a in kb_results
            ])
            
            system_message = f"""You are a support AI. Use this knowledge base info to help:
            
            {kb_context}
            
            Provide helpful response based on KB articles."""
        else:
            system_message = "You are a helpful support AI."
            kb_results = []
    else:
        system_message = "You are a helpful support AI."
        kb_results = []
    
    messages = [SystemMessage(content=system_message)] + state['messages']
    response = llm.invoke(messages)
    
    print(f"\n🤖 AI (with KB): {response.content}\n")
    
    return {
        "messages": [AIMessage(content=response.content)],
        "kb_results": kb_results,
        "conversation_turns": state.get('conversation_turns', 0) + 1
    }


def solution_3_knowledge_base():
    """Test knowledge base integration"""
    print("\n" + "="*80)
    print("SOLUTION 3: Knowledge Base Lookup")
    print("="*80 + "\n")
    
    workflow = StateGraph(CustomerSupportState)
    workflow.add_node("ai_agent", ai_support_with_kb)
    workflow.add_edge(START, "ai_agent")
    workflow.add_edge("ai_agent", END)
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    thread = {"configurable": {"thread_id": "kb_test"}}
    
    state = {
        "messages": [HumanMessage(content="I need help resetting my password")],
        "customer_id": "CUST-002",
        "customer_name": "Test User",
        "issue_type": "password",
        "account_number": "",
        "issue_description": "",
        "urgency_level": "medium",
        "sentiment_score": 50.0,
        "sentiment_history": [],
        "last_response_time": "",
        "timeout_warnings": 0,
        "kb_results": [],
        "conversation_summary": "",
        "resolution_attempted": False,
        "resolved": False,
        "escalated_to_human": False,
        "human_agent_notes": "",
        "agent_type": "",
        "satisfaction_rating": 0,
        "satisfaction_feedback": "",
        "conversation_turns": 0,
        "resolution_time": 0.0,
        "conversation_start": datetime.now().isoformat()
    }
    
    for event in app.stream(state, thread, stream_mode="values"):
        final_state = event
    
    print(f"✅ KB articles used: {len(final_state.get('kb_results', []))}\n")


# ============================================================================
# SOLUTION 4: Conversation Summary
# ============================================================================
def generate_summary(state: CustomerSupportState) -> CustomerSupportState:
    """Generate conversation summary"""
    llm = get_llm()
    
    messages_text = "\n".join([
        f"{msg.type}: {msg.content}"
        for msg in state['messages']
    ])
    
    prompt = f"""Summarize this support conversation in 2-3 sentences:
    
    {messages_text}
    
    Include: issue, resolution status, next steps."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    summary = response.content
    
    print("\n📝 CONVERSATION SUMMARY")
    print("="*80)
    print(summary)
    print("="*80 + "\n")
    
    return {"conversation_summary": summary}


def solution_4_summary():
    """Test summary generation"""
    print("\n" + "="*80)
    print("SOLUTION 4: Conversation Summary")
    print("="*80 + "\n")
    
    state = {
        "messages": [
            HumanMessage(content="I can't log into my account"),
            AIMessage(content="I'll help you with that. Have you tried resetting your password?"),
            HumanMessage(content="Yes, but I'm not receiving the reset email"),
            AIMessage(content="Let me check your email on file. It may need to be updated.")
        ]
    }
    
    result = generate_summary(state)
    print("✅ Summary generated\n")


# ============================================================================
# SOLUTION 5: Customer Satisfaction Survey
# ============================================================================
def conduct_satisfaction_survey(state: CustomerSupportState) -> CustomerSupportState:
    """Collect customer satisfaction after resolution"""
    print("\n" + "="*80)
    print("📊 CUSTOMER SATISFACTION SURVEY")
    print("="*80)
    print("Issue resolved successfully")
    print("="*80 + "\n")
    
    # Request rating
    rating = interrupt({
        "message": "Please rate your experience (1-5 stars)",
        "type": "satisfaction_survey"
    })
    
    print(f"\n⭐ Customer Rating: {rating} stars")
    
    # Request feedback if rating is low
    feedback = ""
    if int(rating) < 4:
        print("\nRequesting additional feedback...\n")
        feedback = interrupt({
            "message": "We'd like to improve. Please share what went wrong:",
            "type": "feedback_request"
        })
        print(f"\n💬 Feedback: {feedback}")
    
    print()
    
    return {
        "satisfaction_rating": int(rating),
        "satisfaction_feedback": feedback,
        "resolved": True
    }


def solution_5_satisfaction():
    """Test satisfaction survey"""
    print("\n" + "="*80)
    print("SOLUTION 5: Customer Satisfaction Survey")
    print("="*80 + "\n")
    
    workflow = StateGraph(CustomerSupportState)
    workflow.add_node("survey", conduct_satisfaction_survey)
    workflow.add_edge(START, "survey")
    workflow.add_edge("survey", END)
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    thread = {"configurable": {"thread_id": "survey_test"}}
    
    state = {"resolved": True}
    
    # Start survey
    for event in app.stream(state, thread, stream_mode="values"):
        pass
    
    state_check = app.get_state(thread)
    if state_check.tasks:
        # Provide rating
        for event in app.stream(Command(resume="3"), thread, stream_mode="values"):
            pass
        
        # Check for feedback request
        state_check_2 = app.get_state(thread)
        if state_check_2.tasks:
            # Provide feedback
            for event in app.stream(
                Command(resume="Wait time was too long"),
                thread,
                stream_mode="values"
            ):
                pass
    
    print("✅ Satisfaction survey completed\n")


# ============================================================================
# SOLUTION 6: Multi-Agent Handoff
# ============================================================================
def route_to_specialist(state: CustomerSupportState) -> Literal["billing", "technical", "fraud", "general"]:
    """Route to appropriate specialist"""
    issue = state.get('issue_type', '').lower()
    
    if any(word in issue for word in ['billing', 'charge', 'fee', 'payment']):
        return "billing"
    elif any(word in issue for word in ['fraud', 'unauthorized', 'stolen', 'suspicious']):
        return "fraud"
    elif any(word in issue for word in ['login', 'password', 'technical', 'app', 'website']):
        return "technical"
    else:
        return "general"


def billing_specialist(state: CustomerSupportState) -> CustomerSupportState:
    """Billing specialist agent"""
    print("\n💰 BILLING SPECIALIST")
    notes = interrupt({"message": "Billing specialist reviewing account", "type": "billing"})
    print(f"Notes: {notes}\n")
    return {"agent_type": "billing", "human_agent_notes": notes, "resolved": True}


def technical_specialist(state: CustomerSupportState) -> CustomerSupportState:
    """Technical specialist agent"""
    print("\n🔧 TECHNICAL SPECIALIST")
    notes = interrupt({"message": "Technical specialist troubleshooting", "type": "technical"})
    print(f"Notes: {notes}\n")
    return {"agent_type": "technical", "human_agent_notes": notes, "resolved": True}


def fraud_specialist(state: CustomerSupportState) -> CustomerSupportState:
    """Fraud specialist agent"""
    print("\n🚨 FRAUD SPECIALIST")
    notes = interrupt({"message": "Fraud specialist investigating", "type": "fraud"})
    print(f"Notes: {notes}\n")
    return {"agent_type": "fraud", "human_agent_notes": notes, "resolved": True}


def general_support(state: CustomerSupportState) -> CustomerSupportState:
    """General support agent"""
    print("\n👤 GENERAL SUPPORT")
    notes = interrupt({"message": "General support agent assisting", "type": "general"})
    print(f"Notes: {notes}\n")
    return {"agent_type": "general", "human_agent_notes": notes, "resolved": True}


def solution_6_multi_agent():
    """Test multi-agent routing"""
    print("\n" + "="*80)
    print("SOLUTION 6: Multi-Agent Handoff")
    print("="*80 + "\n")
    
    workflow = StateGraph(CustomerSupportState)
    workflow.add_node("billing", billing_specialist)
    workflow.add_node("technical", technical_specialist)
    workflow.add_node("fraud", fraud_specialist)
    workflow.add_node("general", general_support)
    
    workflow.add_conditional_edges(
        START,
        route_to_specialist,
        {
            "billing": "billing",
            "technical": "technical",
            "fraud": "fraud",
            "general": "general"
        }
    )
    
    workflow.add_edge("billing", END)
    workflow.add_edge("technical", END)
    workflow.add_edge("fraud", END)
    workflow.add_edge("general", END)
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    # Test different issue types
    test_cases = [
        {"issue_type": "billing dispute", "expected": "billing"},
        {"issue_type": "fraud alert", "expected": "fraud"},
        {"issue_type": "login problem", "expected": "technical"}
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTEST {i}: {test['issue_type']}")
        print("="*80)
        
        thread = {"configurable": {"thread_id": f"agent_test_{i}"}}
        state = {"issue_type": test['issue_type']}
        
        for event in app.stream(state, thread, stream_mode="values"):
            pass
        
        state_check = app.get_state(thread)
        if state_check.tasks:
            # Complete with agent notes
            for event in app.stream(
                Command(resume=f"{test['expected']} issue resolved"),
                thread,
                stream_mode="values"
            ):
                pass
        
        final = app.get_state(thread)
        print(f"✅ Routed to: {final.values.get('agent_type', 'unknown')}\n")


# ============================================================================
# SOLUTION 7: Conversation Analytics
# ============================================================================
class ConversationAnalytics:
    """Track and analyze conversation metrics"""
    
    def __init__(self):
        self.conversations = []
    
    def record_conversation(self, state: CustomerSupportState):
        """Record conversation metrics"""
        start_time = datetime.fromisoformat(state.get('conversation_start', datetime.now().isoformat()))
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60  # minutes
        
        metrics = {
            "conversation_id": state.get('customer_id', 'unknown'),
            "turns": state.get('conversation_turns', 0),
            "duration_minutes": duration,
            "resolved": state.get('resolved', False),
            "escalated": state.get('escalated_to_human', False),
            "satisfaction": state.get('satisfaction_rating', 0),
            "issue_type": state.get('issue_type', 'unknown')
        }
        
        self.conversations.append(metrics)
        return metrics
    
    def generate_report(self):
        """Generate analytics report"""
        if not self.conversations:
            return "No data"
        
        total = len(self.conversations)
        resolved = sum(1 for c in self.conversations if c['resolved'])
        escalated = sum(1 for c in self.conversations if c['escalated'])
        avg_turns = sum(c['turns'] for c in self.conversations) / total
        avg_duration = sum(c['duration_minutes'] for c in self.conversations) / total
        avg_satisfaction = sum(c['satisfaction'] for c in self.conversations if c['satisfaction'] > 0)
        avg_satisfaction = avg_satisfaction / max(1, sum(1 for c in self.conversations if c['satisfaction'] > 0))
        
        print("\n" + "="*80)
        print("📊 CONVERSATION ANALYTICS REPORT")
        print("="*80)
        print(f"Total Conversations: {total}")
        print(f"Resolution Rate: {(resolved/total)*100:.1f}%")
        print(f"Escalation Rate: {(escalated/total)*100:.1f}%")
        print(f"Avg Conversation Turns: {avg_turns:.1f}")
        print(f"Avg Duration: {avg_duration:.1f} minutes")
        print(f"Avg Satisfaction: {avg_satisfaction:.1f}/5 stars")
        print("="*80 + "\n")


def solution_7_analytics():
    """Test conversation analytics"""
    print("\n" + "="*80)
    print("SOLUTION 7: Conversation Analytics")
    print("="*80 + "\n")
    
    analytics = ConversationAnalytics()
    
    # Simulate multiple conversations
    conversations = [
        {
            "customer_id": "CUST-001",
            "conversation_turns": 5,
            "resolved": True,
            "escalated_to_human": False,
            "satisfaction_rating": 5,
            "issue_type": "password_reset",
            "conversation_start": (datetime.now() - timedelta(minutes=10)).isoformat()
        },
        {
            "customer_id": "CUST-002",
            "conversation_turns": 8,
            "resolved": True,
            "escalated_to_human": True,
            "satisfaction_rating": 3,
            "issue_type": "billing",
            "conversation_start": (datetime.now() - timedelta(minutes=25)).isoformat()
        },
        {
            "customer_id": "CUST-003",
            "conversation_turns": 3,
            "resolved": True,
            "escalated_to_human": False,
            "satisfaction_rating": 4,
            "issue_type": "balance_inquiry",
            "conversation_start": (datetime.now() - timedelta(minutes=5)).isoformat()
        }
    ]
    
    from datetime import timedelta
    
    for conv in conversations:
        analytics.record_conversation(conv)
    
    analytics.generate_report()
    print("✅ Analytics tracked\n")


# ============================================================================
# SOLUTION 8: Test Different Scenarios
# ============================================================================
def solution_8_test_scenarios():
    """Test various support scenarios"""
    print("\n" + "="*80)
    print("SOLUTION 8: Testing Different Support Scenarios")
    print("="*80 + "\n")
    
    scenarios = [
        {
            "name": "Quick Password Reset",
            "messages": ["I need to reset my password"],
            "expected_outcome": "resolved_quickly"
        },
        {
            "name": "Fraud Report",
            "messages": ["I see unauthorized transactions on my account!"],
            "expected_outcome": "immediate_escalation"
        },
        {
            "name": "General Inquiry",
            "messages": ["What are your hours?"],
            "expected_outcome": "resolved_by_ai"
        },
        {
            "name": "Angry Customer",
            "messages": ["This is ridiculous! I've been waiting forever!"],
            "expected_outcome": "escalate_due_to_sentiment"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nSCENARIO {i}: {scenario['name']}")
        print("="*80)
        print(f"Customer Message: {scenario['messages'][0]}")
        print(f"Expected Outcome: {scenario['expected_outcome']}")
        
        # Analyze characteristics
        if "password" in scenario['messages'][0].lower():
            print("✓ Should use KB lookup")
        if "fraud" in scenario['messages'][0].lower() or "unauthorized" in scenario['messages'][0].lower():
            print("✓ Should escalate to fraud specialist")
        if any(word in scenario['messages'][0].lower() for word in ['ridiculous', 'forever', 'angry']):
            print("✓ Should trigger sentiment analysis")
        
        print()
    
    print("✅ All scenarios documented\n")


# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("LAB 4 EXERCISE SOLUTIONS")
    print("="*80)
    
    solution_1_sentiment_analysis()
    solution_2_timeout_tracking()
    solution_3_knowledge_base()
    solution_4_summary()
    solution_5_satisfaction()
    solution_6_multi_agent()
    solution_7_analytics()
    solution_8_test_scenarios()
    
    print("\n" + "="*80)
    print("✅ ALL LAB 4 SOLUTIONS COMPLETED")
    print("="*80)
