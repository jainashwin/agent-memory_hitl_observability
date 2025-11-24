"""
Lab 2 Solutions: Fraud Detection - Exercise Solutions
======================================================
This file contains complete solutions for all Lab 2 exercises.

Exercises Covered:
1. Adjust fraud threshold from 60 to 50
2. Add multiple interrupt points (amount-based)
3. Implement three-tier review system
4. Add rejection handling with reason capture
5. Create custom transaction scenarios
6. Implement interrupt validation (multi-interrupt pattern)
"""

import os
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json


# ============================================================================
# State Definition
# ============================================================================
class TransactionState(TypedDict):
    transaction_id: str
    account_holder: str
    transaction_amount: float
    merchant: str
    location: str
    transaction_type: str
    account_balance: float
    recent_transactions: list
    fraud_score: float
    fraud_reason: str
    fraud_analyst_decision: str
    analyst_level: str
    rejection_reason: str
    final_status: str
    messages: Annotated[list, "conversation history"]


# ============================================================================
# LLM Setup
# ============================================================================
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found")
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.2, groq_api_key=api_key)


# ============================================================================
# SOLUTION 1: Adjust Fraud Threshold to 50
# ============================================================================
def analyze_transaction_basic(state: TransactionState) -> TransactionState:
    """Analyze transaction for fraud indicators"""
    llm = get_llm()
    
    system_message = """You are a fraud detection AI.
    Analyze transactions and calculate fraud risk score (0-100).
    
    High-risk indicators:
    - Large amounts (>$5,000)
    - Foreign/unusual locations
    - Multiple transactions in short time
    - Amount > 50% of account balance
    - Unusual merchants
    
    Return ONLY JSON: {"fraud_score": <number>, "fraud_reason": "<text>"}"""
    
    recent_txns = ", ".join([f"${t['amount']} at {t['merchant']}" 
                             for t in state['recent_transactions']])
    
    prompt = f"""
    Transaction: {state['transaction_id']}
    Amount: ${state['transaction_amount']:,.2f}
    Merchant: {state['merchant']}
    Location: {state['location']}
    Balance: ${state['account_balance']:,.2f}
    Recent: {recent_txns}
    
    Analyze and return JSON.
    """
    
    response = llm.invoke([SystemMessage(content=system_message), HumanMessage(content=prompt)])
    
    try:
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        fraud_data = json.loads(content)
        fraud_score = fraud_data.get('fraud_score', 30)
        fraud_reason = fraud_data.get('fraud_reason', 'Analysis completed')
    except:
        fraud_score = 30
        fraud_reason = "Error parsing response"
    
    print(f"\n🔍 Fraud Score: {fraud_score}/100")
    print(f"Reason: {fraud_reason}\n")
    
    return {"fraud_score": fraud_score, "fraud_reason": fraud_reason}


def review_suspicious_threshold_50(state: TransactionState) -> TransactionState:
    """
    SOLUTION 1: Lowered threshold to 50 (from 60)
    This will trigger more transactions for human review
    """
    fraud_score = state['fraud_score']
    
    # CHANGED THRESHOLD: 50 instead of 60
    if fraud_score > 50:  # Lower threshold
        print(f"\n⚠️  MEDIUM-HIGH FRAUD RISK ({fraud_score}) - Pausing for review")
        
        analyst_decision = interrupt({
            "message": "Fraud analyst review required",
            "fraud_score": fraud_score,
            "threshold": 50,  # Document the threshold used
            "amount": state['transaction_amount']
        })
        
        print(f"\n👤 Analyst Decision: {analyst_decision}\n")
        return {"fraud_analyst_decision": analyst_decision, "final_status": analyst_decision}
    else:
        print(f"\n✅ LOW RISK ({fraud_score}) - Auto-approved\n")
        return {"fraud_analyst_decision": "AUTO_APPROVED", "final_status": "APPROVED"}


def process_transaction_basic(state: TransactionState) -> TransactionState:
    """Process transaction"""
    status = state['final_status']
    print(f"\n💳 Transaction {status}\n")
    return {"final_status": f"Transaction {state['transaction_id']} {status}"}


def solution_1_lower_threshold():
    """Test with lower fraud threshold of 50"""
    print("\n" + "="*80)
    print("SOLUTION 1: Lower Fraud Threshold (50 instead of 60)")
    print("="*80 + "\n")
    
    workflow = StateGraph(TransactionState)
    workflow.add_node("analyze_transaction", analyze_transaction_basic)
    workflow.add_node("review_suspicious", review_suspicious_threshold_50)
    workflow.add_node("process_transaction", process_transaction_basic)
    
    workflow.add_edge(START, "analyze_transaction")
    workflow.add_edge("analyze_transaction", "review_suspicious")
    workflow.add_edge("review_suspicious", "process_transaction")
    workflow.add_edge("process_transaction", END)
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    # Test with medium-risk transaction (score likely 50-60)
    thread = {"configurable": {"thread_id": "fraud_sol1"}}
    
    state = {
        "transaction_id": "TXN-SOL1",
        "account_holder": "Test User",
        "transaction_amount": 3500.0,  # Moderately large
        "merchant": "Online Electronics",
        "location": "California, USA",
        "transaction_type": "Purchase",
        "account_balance": 8000.0,
        "recent_transactions": [
            {"amount": 150, "merchant": "Grocery"},
            {"amount": 200, "merchant": "Gas Station"}
        ],
        "fraud_score": 0.0,
        "fraud_reason": "",
        "fraud_analyst_decision": "",
        "analyst_level": "",
        "rejection_reason": "",
        "final_status": "",
        "messages": []
    }
    
    for event in app.stream(state, thread, stream_mode="values"):
        pass
    
    state_check = app.get_state(thread)
    if state_check.tasks:
        print("⏸️  Interrupted - More transactions require review with lower threshold")
        print(f"Score: {state_check.values['fraud_score']}")
        
        # Approve
        for event in app.stream(Command(resume="APPROVED"), thread, stream_mode="values"):
            pass
    
    print("✅ Lower threshold increases human review frequency\n")


# ============================================================================
# SOLUTION 2: Multiple Interrupt Points (Amount-Based)
# ============================================================================
def review_with_multiple_checks(state: TransactionState) -> TransactionState:
    """
    SOLUTION 2: Multiple interrupt points based on different conditions
    - First interrupt if amount > $10,000 (regardless of fraud score)
    - Second interrupt if fraud score > 60
    """
    amount = state['transaction_amount']
    fraud_score = state['fraud_score']
    
    # FIRST CHECK: Large transaction amount
    if amount > 10000:
        print(f"\n💰 LARGE TRANSACTION ALERT (${amount:,.2f})")
        
        large_amount_approval = interrupt({
            "message": "Large transaction requires approval",
            "amount": amount,
            "check_type": "amount_threshold"
        })
        
        print(f"👤 Large Amount Review: {large_amount_approval}")
        
        # If rejected at this stage, stop here
        if large_amount_approval == "REJECTED":
            return {
                "fraud_analyst_decision": large_amount_approval,
                "final_status": "REJECTED",
                "rejection_reason": "Large amount rejected by reviewer"
            }
    
    # SECOND CHECK: High fraud score
    if fraud_score > 60:
        print(f"\n⚠️  HIGH FRAUD RISK ({fraud_score})")
        
        fraud_review_decision = interrupt({
            "message": "High fraud score requires review",
            "fraud_score": fraud_score,
            "check_type": "fraud_score"
        })
        
        print(f"👤 Fraud Review: {fraud_review_decision}")
        return {
            "fraud_analyst_decision": fraud_review_decision,
            "final_status": fraud_review_decision
        }
    
    # Both checks passed
    print(f"\n✅ All checks passed - Auto-approved\n")
    return {
        "fraud_analyst_decision": "AUTO_APPROVED",
        "final_status": "APPROVED"
    }


def solution_2_multiple_interrupts():
    """Test multiple interrupt points"""
    print("\n" + "="*80)
    print("SOLUTION 2: Multiple Interrupt Points")
    print("="*80 + "\n")
    
    workflow = StateGraph(TransactionState)
    workflow.add_node("analyze_transaction", analyze_transaction_basic)
    workflow.add_node("review_suspicious", review_with_multiple_checks)
    workflow.add_node("process_transaction", process_transaction_basic)
    
    workflow.add_edge(START, "analyze_transaction")
    workflow.add_edge("analyze_transaction", "review_suspicious")
    workflow.add_edge("review_suspicious", "process_transaction")
    workflow.add_edge("process_transaction", END)
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    # Test with large amount transaction
    thread = {"configurable": {"thread_id": "fraud_sol2"}}
    
    state = {
        "transaction_id": "TXN-SOL2",
        "account_holder": "High Value Customer",
        "transaction_amount": 15000.0,  # Triggers amount check
        "merchant": "Jewelry Store",
        "location": "New York, USA",
        "transaction_type": "Purchase",
        "account_balance": 50000.0,
        "recent_transactions": [{"amount": 200, "merchant": "Restaurant"}],
        "fraud_score": 0.0,
        "fraud_reason": "",
        "fraud_analyst_decision": "",
        "analyst_level": "",
        "rejection_reason": "",
        "final_status": "",
        "messages": []
    }
    
    for event in app.stream(state, thread, stream_mode="values"):
        pass
    
    state_check = app.get_state(thread)
    if state_check.tasks:
        interrupt_data = state_check.tasks[0].interrupts[0].value
        print(f"\n⏸️  First Interrupt: {interrupt_data['check_type']}")
        
        # Approve amount check
        for event in app.stream(Command(resume="APPROVED"), thread, stream_mode="values"):
            pass
        
        # Check for second interrupt
        state_check_2 = app.get_state(thread)
        if state_check_2.tasks:
            interrupt_data_2 = state_check_2.tasks[0].interrupts[0].value
            print(f"\n⏸️  Second Interrupt: {interrupt_data_2['check_type']}")
            
            # Approve fraud check
            for event in app.stream(Command(resume="APPROVED"), thread, stream_mode="values"):
                pass
    
    print("✅ Multiple interrupt points tested\n")


# ============================================================================
# SOLUTION 3: Three-Tier Review System
# ============================================================================
def review_three_tier(state: TransactionState) -> TransactionState:
    """
    SOLUTION 3: Three-tier review system
    - Low risk (0-40): Auto-approve
    - Medium risk (41-70): Junior analyst review
    - High risk (71-100): Senior analyst review
    """
    fraud_score = state['fraud_score']
    
    if fraud_score <= 40:
        # LOW RISK - Auto approve
        print(f"\n✅ LOW RISK ({fraud_score}) - Auto-approved\n")
        return {
            "fraud_analyst_decision": "AUTO_APPROVED",
            "analyst_level": "none",
            "final_status": "APPROVED"
        }
    
    elif fraud_score <= 70:
        # MEDIUM RISK - Junior analyst
        print(f"\n⚠️  MEDIUM RISK ({fraud_score}) - Junior Analyst Review Required")
        
        decision = interrupt({
            "message": "Junior analyst review required",
            "fraud_score": fraud_score,
            "analyst_level": "junior",
            "transaction_id": state['transaction_id']
        })
        
        print(f"👤 Junior Analyst Decision: {decision}\n")
        return {
            "fraud_analyst_decision": decision,
            "analyst_level": "junior",
            "final_status": decision
        }
    
    else:
        # HIGH RISK - Senior analyst
        print(f"\n🚨 HIGH RISK ({fraud_score}) - Senior Analyst Review Required")
        
        decision = interrupt({
            "message": "Senior analyst review required",
            "fraud_score": fraud_score,
            "analyst_level": "senior",
            "transaction_id": state['transaction_id']
        })
        
        print(f"👤 Senior Analyst Decision: {decision}\n")
        return {
            "fraud_analyst_decision": decision,
            "analyst_level": "senior",
            "final_status": decision
        }


def solution_3_three_tier():
    """Test three-tier review system"""
    print("\n" + "="*80)
    print("SOLUTION 3: Three-Tier Review System")
    print("="*80 + "\n")
    
    workflow = StateGraph(TransactionState)
    workflow.add_node("analyze_transaction", analyze_transaction_basic)
    workflow.add_node("review_suspicious", review_three_tier)
    workflow.add_node("process_transaction", process_transaction_basic)
    
    workflow.add_edge(START, "analyze_transaction")
    workflow.add_edge("analyze_transaction", "review_suspicious")
    workflow.add_edge("review_suspicious", "process_transaction")
    workflow.add_edge("process_transaction", END)
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    # Test all three tiers
    test_cases = [
        {
            "name": "Low Risk (Auto-approve)",
            "transaction_amount": 50.0,
            "merchant": "Coffee Shop",
            "location": "Local"
        },
        {
            "name": "Medium Risk (Junior Analyst)",
            "transaction_amount": 2000.0,
            "merchant": "Electronics Store - Different City",
            "location": "Out of State"
        },
        {
            "name": "High Risk (Senior Analyst)",
            "transaction_amount": 9000.0,
            "merchant": "Wire Transfer Service - Foreign",
            "location": "International"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}: {test['name']}")
        print(f"{'='*80}\n")
        
        thread = {"configurable": {"thread_id": f"fraud_sol3_{i}"}}
        
        state = {
            "transaction_id": f"TXN-TIER-{i}",
            "account_holder": "Test User",
            "transaction_amount": test['transaction_amount'],
            "merchant": test['merchant'],
            "location": test['location'],
            "transaction_type": "Purchase",
            "account_balance": 10000.0,
            "recent_transactions": [{"amount": 100, "merchant": "Grocery"}],
            "fraud_score": 0.0,
            "fraud_reason": "",
            "fraud_analyst_decision": "",
            "analyst_level": "",
            "rejection_reason": "",
            "final_status": "",
            "messages": []
        }
        
        for event in app.stream(state, thread, stream_mode="values"):
            pass
        
        state_check = app.get_state(thread)
        if state_check.tasks:
            interrupt_data = state_check.tasks[0].interrupts[0].value
            print(f"Analyst Level Required: {interrupt_data['analyst_level'].upper()}")
            
            # Approve
            for event in app.stream(Command(resume="APPROVED"), thread, stream_mode="values"):
                pass
    
    print("\n✅ Three-tier system tested\n")


# ============================================================================
# SOLUTION 4: Rejection with Reason Capture
# ============================================================================
def review_with_rejection(state: TransactionState) -> TransactionState:
    """
    SOLUTION 4: When analyst rejects, ask for detailed reason
    Demonstrates multiple sequential interrupts
    """
    fraud_score = state['fraud_score']
    
    if fraud_score > 60:
        print(f"\n⚠️  HIGH FRAUD RISK ({fraud_score}) - Review Required")
        
        # First interrupt: Approve or Reject
        decision = interrupt({
            "message": "Review transaction - APPROVE or REJECT",
            "fraud_score": fraud_score
        })
        
        print(f"👤 Analyst Decision: {decision}")
        
        # If rejected, get detailed reason
        if decision == "REJECTED":
            print("\n📝 Rejection detected - Requesting detailed reason")
            
            # Second interrupt: Get rejection reason
            rejection_reason = interrupt({
                "message": "Please provide detailed rejection reason",
                "transaction_id": state['transaction_id']
            })
            
            print(f"👤 Rejection Reason: {rejection_reason}\n")
            
            return {
                "fraud_analyst_decision": decision,
                "final_status": "REJECTED",
                "rejection_reason": rejection_reason
            }
        else:
            return {
                "fraud_analyst_decision": decision,
                "final_status": "APPROVED",
                "rejection_reason": ""
            }
    else:
        return {
            "fraud_analyst_decision": "AUTO_APPROVED",
            "final_status": "APPROVED",
            "rejection_reason": ""
        }


def solution_4_rejection_reason():
    """Test rejection with reason capture"""
    print("\n" + "="*80)
    print("SOLUTION 4: Rejection with Detailed Reason")
    print("="*80 + "\n")
    
    workflow = StateGraph(TransactionState)
    workflow.add_node("analyze_transaction", analyze_transaction_basic)
    workflow.add_node("review_suspicious", review_with_rejection)
    workflow.add_node("process_transaction", process_transaction_basic)
    
    workflow.add_edge(START, "analyze_transaction")
    workflow.add_edge("analyze_transaction", "review_suspicious")
    workflow.add_edge("review_suspicious", "process_transaction")
    workflow.add_edge("process_transaction", END)
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    thread = {"configurable": {"thread_id": "fraud_sol4"}}
    
    state = {
        "transaction_id": "TXN-SOL4",
        "account_holder": "Suspicious Account",
        "transaction_amount": 7500.0,
        "merchant": "Cryptocurrency Exchange - Foreign",
        "location": "Unknown Location",
        "transaction_type": "Wire Transfer",
        "account_balance": 8000.0,
        "recent_transactions": [
            {"amount": 5000, "merchant": "Foreign Wire"},
            {"amount": 4500, "merchant": "Crypto Exchange"}
        ],
        "fraud_score": 0.0,
        "fraud_reason": "",
        "fraud_analyst_decision": "",
        "analyst_level": "",
        "rejection_reason": "",
        "final_status": "",
        "messages": []
    }
    
    for event in app.stream(state, thread, stream_mode="values"):
        pass
    
    # First interrupt - decision
    state_check = app.get_state(thread)
    if state_check.tasks:
        print("\n⏸️  First Interrupt: Approve or Reject")
        
        # Analyst rejects
        for event in app.stream(Command(resume="REJECTED"), thread, stream_mode="values"):
            pass
        
        # Second interrupt - reason
        state_check_2 = app.get_state(thread)
        if state_check_2.tasks:
            print("\n⏸️  Second Interrupt: Provide Rejection Reason")
            
            reason = "Pattern of high-value cryptocurrency transactions to unverified exchanges. " \
                    "Multiple red flags: foreign location, rapid transaction velocity, " \
                    "and account behavior inconsistent with profile."
            
            for event in app.stream(Command(resume=reason), thread, stream_mode="values"):
                pass
    
    # Check final state
    final = app.get_state(thread)
    print(f"\n📊 Final Status: {final.values['final_status']}")
    print(f"📝 Rejection Reason: {final.values['rejection_reason']}")
    
    print("\n✅ Multi-interrupt rejection pattern tested\n")


# ============================================================================
# SOLUTION 5: Custom Transaction Scenarios
# ============================================================================
def solution_5_custom_scenarios():
    """
    SOLUTION 5: Test various custom transaction scenarios
    - International wire transfer
    - ATM withdrawal in foreign country
    - Structuring pattern (multiple small transactions)
    """
    print("\n" + "="*80)
    print("SOLUTION 5: Custom Transaction Scenarios")
    print("="*80 + "\n")
    
    workflow = StateGraph(TransactionState)
    workflow.add_node("analyze_transaction", analyze_transaction_basic)
    workflow.add_node("review_suspicious", review_suspicious_threshold_50)
    workflow.add_node("process_transaction", process_transaction_basic)
    
    workflow.add_edge(START, "analyze_transaction")
    workflow.add_edge("analyze_transaction", "review_suspicious")
    workflow.add_edge("review_suspicious", "process_transaction")
    workflow.add_edge("process_transaction", END)
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    scenarios = [
        {
            "name": "International Wire Transfer",
            "transaction_id": "TXN-INTL-WIRE",
            "amount": 25000.0,
            "merchant": "International Wire Transfer Service",
            "location": "Switzerland",
            "type": "Wire Transfer",
            "recent": [
                {"amount": 100, "merchant": "Local Grocery"},
                {"amount": 50, "merchant": "Gas Station"}
            ]
        },
        {
            "name": "Foreign ATM Withdrawal",
            "transaction_id": "TXN-FOREIGN-ATM",
            "amount": 500.0,
            "merchant": "ATM Withdrawal",
            "location": "Bangkok, Thailand",
            "type": "ATM Withdrawal",
            "recent": [
                {"amount": 45, "merchant": "Coffee Shop - USA"},
                {"amount": 120, "merchant": "Restaurant - USA"}
            ]
        },
        {
            "name": "Structuring Pattern",
            "transaction_id": "TXN-STRUCTURING",
            "amount": 9500.0,
            "merchant": "Check Cashing Service",
            "location": "Local",
            "type": "Cash Deposit",
            "recent": [
                {"amount": 9800, "merchant": "Check Cashing"},
                {"amount": 9600, "merchant": "Money Order"},
                {"amount": 9700, "merchant": "Check Cashing"},
                {"amount": 9400, "merchant": "Wire Transfer"}
            ]
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*80}")
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"{'='*80}\n")
        
        thread = {"configurable": {"thread_id": f"custom_scenario_{i}"}}
        
        state = {
            "transaction_id": scenario['transaction_id'],
            "account_holder": "Test Customer",
            "transaction_amount": scenario['amount'],
            "merchant": scenario['merchant'],
            "location": scenario['location'],
            "transaction_type": scenario['type'],
            "account_balance": 50000.0,
            "recent_transactions": scenario['recent'],
            "fraud_score": 0.0,
            "fraud_reason": "",
            "fraud_analyst_decision": "",
            "analyst_level": "",
            "rejection_reason": "",
            "final_status": "",
            "messages": []
        }
        
        for event in app.stream(state, thread, stream_mode="values"):
            pass
        
        state_check = app.get_state(thread)
        if state_check.tasks:
            print(f"⏸️  Human review triggered")
            fraud_score = state_check.values['fraud_score']
            print(f"Fraud Score: {fraud_score}")
            
            # Auto-approve for demo
            for event in app.stream(Command(resume="APPROVED"), thread, stream_mode="values"):
                pass
        
        print(f"✅ {scenario['name']} processed")
    
    print("\n✅ All custom scenarios tested\n")


# ============================================================================
# SOLUTION 6: Interrupt Validation (Multi-Interrupt Pattern)
# ============================================================================
def review_with_validation(state: TransactionState) -> TransactionState:
    """
    SOLUTION 6: Validate analyst input and re-prompt if invalid
    Demonstrates validation and re-prompting pattern
    """
    fraud_score = state['fraud_score']
    
    if fraud_score > 60:
        print(f"\n⚠️  HIGH FRAUD RISK ({fraud_score})")
        
        # Keep asking until we get valid input
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            
            decision = interrupt({
                "message": f"Review required (Attempt {attempt}/{max_attempts})",
                "valid_options": ["APPROVED", "REJECTED"],
                "fraud_score": fraud_score
            })
            
            # Validate input
            if decision.upper() in ["APPROVED", "REJECTED"]:
                print(f"✅ Valid decision received: {decision}\n")
                return {
                    "fraud_analyst_decision": decision.upper(),
                    "final_status": decision.upper()
                }
            else:
                print(f"❌ Invalid input: '{decision}'")
                print(f"Please enter APPROVED or REJECTED\n")
        
        # Max attempts reached
        print("⚠️  Max attempts reached - Escalating to supervisor")
        return {
            "fraud_analyst_decision": "ESCALATED",
            "final_status": "ESCALATED"
        }
    else:
        return {
            "fraud_analyst_decision": "AUTO_APPROVED",
            "final_status": "APPROVED"
        }


def solution_6_validation():
    """Test interrupt validation pattern"""
    print("\n" + "="*80)
    print("SOLUTION 6: Interrupt Validation Pattern")
    print("="*80 + "\n")
    
    workflow = StateGraph(TransactionState)
    workflow.add_node("analyze_transaction", analyze_transaction_basic)
    workflow.add_node("review_suspicious", review_with_validation)
    workflow.add_node("process_transaction", process_transaction_basic)
    
    workflow.add_edge(START, "analyze_transaction")
    workflow.add_edge("analyze_transaction", "review_suspicious")
    workflow.add_edge("review_suspicious", "process_transaction")
    workflow.add_edge("process_transaction", END)
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    thread = {"configurable": {"thread_id": "fraud_sol6"}}
    
    state = {
        "transaction_id": "TXN-SOL6",
        "account_holder": "Test User",
        "transaction_amount": 8000.0,
        "merchant": "High Risk Merchant",
        "location": "Foreign",
        "transaction_type": "Purchase",
        "account_balance": 10000.0,
        "recent_transactions": [{"amount": 100, "merchant": "Normal"}],
        "fraud_score": 0.0,
        "fraud_reason": "",
        "fraud_analyst_decision": "",
        "analyst_level": "",
        "rejection_reason": "",
        "final_status": "",
        "messages": []
    }
    
    for event in app.stream(state, thread, stream_mode="values"):
        pass
    
    # Simulate invalid then valid input
    state_check = app.get_state(thread)
    if state_check.tasks:
        print("⏸️  First attempt with invalid input")
        
        # Invalid input
        for event in app.stream(Command(resume="MAYBE"), thread, stream_mode="values"):
            pass
        
        state_check_2 = app.get_state(thread)
        if state_check_2.tasks:
            print("⏸️  Second attempt with valid input")
            
            # Valid input
            for event in app.stream(Command(resume="APPROVED"), thread, stream_mode="values"):
                pass
    
    print("✅ Validation pattern tested\n")


# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("LAB 2 EXERCISE SOLUTIONS")
    print("="*80)
    
    solution_1_lower_threshold()
    solution_2_multiple_interrupts()
    solution_3_three_tier()
    solution_4_rejection_reason()
    solution_5_custom_scenarios()
    solution_6_validation()
    
    print("\n" + "="*80)
    print("✅ ALL LAB 2 SOLUTIONS COMPLETED")
    print("="*80)
