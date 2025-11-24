"""
Lab 3 Solutions: Account Opening - Exercise Solutions
======================================================
This file contains complete solutions for all Lab 3 exercises.

Exercises Covered:
1. Edit multiple fields simultaneously
2. Add approval step with manager review
3. Implement correction loop with retry logic
4. Create validation history tracking
5. Build interactive review interface
6. Add compliance checks (age, deposit limits)
7. Test incomplete applications
"""

import os
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import json
from datetime import datetime


# ============================================================================
# State Definition
# ============================================================================
class AccountOpeningState(TypedDict):
    # Raw data
    raw_application: str
    
    # Extracted data
    customer_name: str
    date_of_birth: str
    ssn_last_four: str
    address: str
    phone: str
    email: str
    initial_deposit: float
    account_type: str
    
    # Validation
    data_validated: bool
    validation_errors: list
    retry_count: int
    validation_history: list
    
    # Compliance
    age_verified: bool
    deposit_limit_ok: bool
    compliance_approved: bool
    
    # Manager approval
    manager_approved: bool
    manager_notes: str
    
    # Final
    account_number: str
    final_status: str
    messages: Annotated[list, "conversation history"]


# ============================================================================
# LLM Setup
# ============================================================================
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found")
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0, groq_api_key=api_key)


# ============================================================================
# Core Functions
# ============================================================================
def extract_customer_data(state: AccountOpeningState) -> AccountOpeningState:
    """Extract structured data from raw application"""
    llm = get_llm()
    
    system_message = """Extract customer information from application text.
    Return ONLY JSON with: customer_name, date_of_birth (YYYY-MM-DD), 
    ssn_last_four, address, phone (XXX-XXX-XXXX), email, initial_deposit, 
    account_type (checking/savings). Use "NEEDS_REVIEW" for missing/unclear fields."""
    
    prompt = f"Extract data from:\n\n{state['raw_application']}\n\nReturn JSON only."
    
    response = llm.invoke([SystemMessage(content=system_message), HumanMessage(content=prompt)])
    
    try:
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        data = json.loads(content)
    except:
        data = {
            "customer_name": "NEEDS_REVIEW",
            "date_of_birth": "NEEDS_REVIEW",
            "ssn_last_four": "NEEDS_REVIEW",
            "address": "NEEDS_REVIEW",
            "phone": "NEEDS_REVIEW",
            "email": "NEEDS_REVIEW",
            "initial_deposit": 0.0,
            "account_type": "NEEDS_REVIEW"
        }
    
    print("\n" + "="*80)
    print("🤖 DATA EXTRACTION")
    print("="*80)
    print(json.dumps(data, indent=2))
    print("="*80 + "\n")
    
    return {
        "customer_name": data.get("customer_name", "NEEDS_REVIEW"),
        "date_of_birth": data.get("date_of_birth", "NEEDS_REVIEW"),
        "ssn_last_four": data.get("ssn_last_four", "NEEDS_REVIEW"),
        "address": data.get("address", "NEEDS_REVIEW"),
        "phone": data.get("phone", "NEEDS_REVIEW"),
        "email": data.get("email", "NEEDS_REVIEW"),
        "initial_deposit": float(data.get("initial_deposit", 0)),
        "account_type": data.get("account_type", "NEEDS_REVIEW")
    }


def validate_customer_data(state: AccountOpeningState) -> AccountOpeningState:
    """Validate extracted data"""
    errors = []
    
    if "NEEDS_REVIEW" in state.get('customer_name', ''):
        errors.append("Customer name needs review")
    
    if "NEEDS_REVIEW" in state.get('date_of_birth', ''):
        errors.append("Date of birth needs review")
    
    ssn = state.get('ssn_last_four', '')
    if ssn == "NEEDS_REVIEW" or not ssn.isdigit() or len(ssn) != 4:
        errors.append("SSN last four digits invalid")
    
    if "NEEDS_REVIEW" in state.get('address', ''):
        errors.append("Address needs review")
    
    email = state.get('email', '')
    if "NEEDS_REVIEW" in email or '@' not in email:
        errors.append("Email address invalid")
    
    if state.get('initial_deposit', 0) < 100:
        errors.append("Initial deposit must be at least $100")
    
    if state.get('account_type', '') not in ['checking', 'savings']:
        errors.append("Account type must be checking or savings")
    
    is_valid = len(errors) == 0
    
    # Add to validation history
    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "attempt": state.get('retry_count', 0) + 1,
        "valid": is_valid,
        "errors": errors
    }
    
    validation_history = state.get('validation_history', [])
    validation_history.append(history_entry)
    
    print("\n" + "="*80)
    print("✓ DATA VALIDATION")
    print("="*80)
    if is_valid:
        print("✅ All data validated successfully")
    else:
        print("⚠️  Validation errors:")
        for error in errors:
            print(f"   - {error}")
    print("="*80 + "\n")
    
    return {
        "data_validated": is_valid,
        "validation_errors": errors,
        "validation_history": validation_history
    }


# ============================================================================
# SOLUTION 1: Edit Multiple Fields Simultaneously
# ============================================================================
def solution_1_multi_field_edit():
    """Edit multiple fields at once using update_state"""
    print("\n" + "="*80)
    print("SOLUTION 1: Edit Multiple Fields Simultaneously")
    print("="*80 + "\n")
    
    workflow = StateGraph(AccountOpeningState)
    workflow.add_node("extract_data", extract_customer_data)
    workflow.add_node("validate_data", validate_customer_data)
    
    workflow.add_edge(START, "extract_data")
    workflow.add_edge("extract_data", "validate_data")
    workflow.add_edge("validate_data", END)
    
    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory,
        interrupt_after=["validate_data"]
    )
    
    thread = {"configurable": {"thread_id": "account_sol1"}}
    
    # Application with multiple errors
    state = {
        "raw_application": """
        I want to open an account. My name is John Doe.
        I was born in 1990. My SSN ends in 1234.
        I live at 123 Main St. 
        Email: john@email.com
        I'll deposit $150.
        """,
        "customer_name": "",
        "date_of_birth": "",
        "ssn_last_four": "",
        "address": "",
        "phone": "",
        "email": "",
        "initial_deposit": 0.0,
        "account_type": "",
        "data_validated": False,
        "validation_errors": [],
        "retry_count": 0,
        "validation_history": [],
        "age_verified": False,
        "deposit_limit_ok": False,
        "compliance_approved": False,
        "manager_approved": False,
        "manager_notes": "",
        "account_number": "",
        "final_status": "",
        "messages": []
    }
    
    for event in app.stream(state, thread, stream_mode="values"):
        pass
    
    current = app.get_state(thread)
    print("\n📊 Current State:")
    print(f"Name: {current.values.get('customer_name')}")
    print(f"DOB: {current.values.get('date_of_birth')}")
    print(f"Phone: {current.values.get('phone')}")
    print(f"Account Type: {current.values.get('account_type')}")
    print(f"Errors: {current.values.get('validation_errors')}\n")
    
    # SOLUTION: Edit multiple fields at once
    print("👤 Representative correcting multiple fields...\n")
    
    corrections = {
        "customer_name": "John A. Doe",
        "date_of_birth": "1990-05-15",
        "phone": "555-123-4567",
        "account_type": "checking",
        "address": "123 Main Street, Apt 4B, Springfield, IL 62701",
        "data_validated": True,
        "validation_errors": []
    }
    
    app.update_state(thread, corrections)
    
    updated = app.get_state(thread)
    print("✅ Multiple fields updated:")
    for key, value in corrections.items():
        if key not in ['data_validated', 'validation_errors']:
            print(f"   - {key}: {value}")
    
    print("\n✅ Multi-field editing demonstrated\n")


# ============================================================================
# SOLUTION 2: Add Manager Approval Step
# ============================================================================
def manager_review(state: AccountOpeningState) -> AccountOpeningState:
    """Manager reviews and approves account opening"""
    print("\n" + "="*80)
    print("👔 MANAGER REVIEW")
    print("="*80)
    print(f"Customer: {state['customer_name']}")
    print(f"Account Type: {state['account_type']}")
    print(f"Initial Deposit: ${state['initial_deposit']:,.2f}")
    print("="*80 + "\n")
    
    return {}  # Manager will update via update_state


def open_account(state: AccountOpeningState) -> AccountOpeningState:
    """Open the account"""
    import random
    account_number = f"ACC-{random.randint(10000000, 99999999)}"
    
    print("\n" + "="*80)
    print("✅ ACCOUNT OPENED")
    print("="*80)
    print(f"Account Number: {account_number}")
    print(f"Customer: {state['customer_name']}")
    print(f"Type: {state['account_type'].upper()}")
    print("="*80 + "\n")
    
    return {"account_number": account_number, "final_status": "OPENED"}


def solution_2_manager_approval():
    """Add manager approval stage"""
    print("\n" + "="*80)
    print("SOLUTION 2: Manager Approval Step")
    print("="*80 + "\n")
    
    workflow = StateGraph(AccountOpeningState)
    workflow.add_node("extract_data", extract_customer_data)
    workflow.add_node("validate_data", validate_customer_data)
    workflow.add_node("manager_review", manager_review)
    workflow.add_node("open_account", open_account)
    
    workflow.add_edge(START, "extract_data")
    workflow.add_edge("extract_data", "validate_data")
    workflow.add_edge("validate_data", "manager_review")
    workflow.add_edge("manager_review", "open_account")
    workflow.add_edge("open_account", END)
    
    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=["manager_review"]  # Pause before manager review
    )
    
    thread = {"configurable": {"thread_id": "account_sol2"}}
    
    state = {
        "raw_application": """
        New account application for Sarah Johnson
        Born: March 10, 1985
        SSN: XXXX-5678
        Address: 456 Oak Avenue, Boston, MA 02101
        Phone: 617-555-0123
        Email: sarah.j@email.com
        Opening deposit: $5,000
        Account type: Savings
        """,
        "customer_name": "",
        "date_of_birth": "",
        "ssn_last_four": "",
        "address": "",
        "phone": "",
        "email": "",
        "initial_deposit": 0.0,
        "account_type": "",
        "data_validated": False,
        "validation_errors": [],
        "retry_count": 0,
        "validation_history": [],
        "age_verified": False,
        "deposit_limit_ok": False,
        "compliance_approved": False,
        "manager_approved": False,
        "manager_notes": "",
        "account_number": "",
        "final_status": "",
        "messages": []
    }
    
    # Run to manager review
    for event in app.stream(state, thread, stream_mode="values"):
        pass
    
    print("\n⏸️  Paused at manager review")
    
    # Manager reviews and approves
    manager_notes = "Reviewed and approved. High-value customer, standard processing."
    app.update_state(thread, {
        "manager_approved": True,
        "manager_notes": manager_notes,
        "data_validated": True
    })
    
    print(f"👔 Manager: {manager_notes}\n")
    
    # Complete
    for event in app.stream(None, thread, stream_mode="values"):
        pass
    
    print("✅ Manager approval workflow completed\n")


# ============================================================================
# SOLUTION 3: Correction Loop with Retry Logic
# ============================================================================
def route_after_validation(state: AccountOpeningState) -> Literal["extract_data", "manager_review", "end"]:
    """Route based on validation results"""
    is_valid = state.get('data_validated', False)
    retry_count = state.get('retry_count', 0)
    
    if is_valid:
        return "manager_review"
    elif retry_count < 3:
        print(f"\n🔄 Validation failed - Retry {retry_count + 1}/3\n")
        return "extract_data"
    else:
        print(f"\n❌ Max retries reached - Manual review required\n")
        return "end"


def increment_retry(state: AccountOpeningState) -> AccountOpeningState:
    """Increment retry counter"""
    return {"retry_count": state.get('retry_count', 0) + 1}


def solution_3_correction_loop():
    """Implement validation retry loop"""
    print("\n" + "="*80)
    print("SOLUTION 3: Correction Loop with Retry Logic")
    print("="*80 + "\n")
    
    workflow = StateGraph(AccountOpeningState)
    workflow.add_node("extract_data", extract_customer_data)
    workflow.add_node("validate_data", validate_customer_data)
    workflow.add_node("increment_retry", increment_retry)
    workflow.add_node("manager_review", manager_review)
    workflow.add_node("open_account", open_account)
    
    workflow.add_edge(START, "extract_data")
    workflow.add_edge("extract_data", "validate_data")
    
    # Conditional routing after validation
    workflow.add_conditional_edges(
        "validate_data",
        route_after_validation,
        {
            "extract_data": "increment_retry",
            "manager_review": "manager_review",
            "end": END
        }
    )
    
    workflow.add_edge("increment_retry", "extract_data")
    workflow.add_edge("manager_review", "open_account")
    workflow.add_edge("open_account", END)
    
    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory,
        interrupt_after=["validate_data"]
    )
    
    thread = {"configurable": {"thread_id": "account_sol3"}}
    
    # Incomplete application (will need corrections)
    state = {
        "raw_application": "Hi, I'm Mike. I want an account.",
        "customer_name": "",
        "date_of_birth": "",
        "ssn_last_four": "",
        "address": "",
        "phone": "",
        "email": "",
        "initial_deposit": 0.0,
        "account_type": "",
        "data_validated": False,
        "validation_errors": [],
        "retry_count": 0,
        "validation_history": [],
        "age_verified": False,
        "deposit_limit_ok": False,
        "compliance_approved": False,
        "manager_approved": False,
        "manager_notes": "",
        "account_number": "",
        "final_status": "",
        "messages": []
    }
    
    # First attempt
    for event in app.stream(state, thread, stream_mode="values"):
        pass
    
    current = app.get_state(thread)
    print(f"Validation Errors: {current.values.get('validation_errors')}")
    print(f"Retry Count: {current.values.get('retry_count')}\n")
    
    # Fix and revalidate
    print("👤 Representative adding missing information...\n")
    app.update_state(thread, {
        "customer_name": "Mike Anderson",
        "date_of_birth": "1988-07-20",
        "ssn_last_four": "9876",
        "address": "789 Elm St, Denver, CO 80202",
        "phone": "303-555-7890",
        "email": "mike.a@email.com",
        "initial_deposit": 500.0,
        "account_type": "checking",
        "data_validated": True,
        "validation_errors": []
    })
    
    # Continue
    for event in app.stream(None, thread, stream_mode="values"):
        pass
    
    final = app.get_state(thread)
    print(f"✅ Final Retry Count: {final.values.get('retry_count')}")
    print(f"✅ Validation History: {len(final.values.get('validation_history', []))} attempts\n")


# ============================================================================
# SOLUTION 4: Validation History Tracking
# ============================================================================
def solution_4_validation_history():
    """Track and display validation history"""
    print("\n" + "="*80)
    print("SOLUTION 4: Validation History Tracking")
    print("="*80 + "\n")
    
    workflow = StateGraph(AccountOpeningState)
    workflow.add_node("extract_data", extract_customer_data)
    workflow.add_node("validate_data", validate_customer_data)
    workflow.add_node("open_account", open_account)
    
    workflow.add_edge(START, "extract_data")
    workflow.add_edge("extract_data", "validate_data")
    workflow.add_edge("validate_data", "open_account")
    workflow.add_edge("open_account", END)
    
    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory,
        interrupt_after=["validate_data"]
    )
    
    thread = {"configurable": {"thread_id": "account_sol4"}}
    
    state = {
        "raw_application": "Tom Smith, born 1995, SSN: 4321, email: tom@email",
        "customer_name": "",
        "date_of_birth": "",
        "ssn_last_four": "",
        "address": "",
        "phone": "",
        "email": "",
        "initial_deposit": 0.0,
        "account_type": "",
        "data_validated": False,
        "validation_errors": [],
        "retry_count": 0,
        "validation_history": [],
        "age_verified": False,
        "deposit_limit_ok": False,
        "compliance_approved": False,
        "manager_approved": False,
        "manager_notes": "",
        "account_number": "",
        "final_status": "",
        "messages": []
    }
    
    # First validation
    for event in app.stream(state, thread, stream_mode="values"):
        pass
    
    # Fix some fields
    app.update_state(thread, {
        "address": "101 Pine St",
        "phone": "555-0101",
        "email": "tom@email.com",
        "initial_deposit": 200.0,
        "account_type": "savings",
        "date_of_birth": "1995-06-12"
    })
    
    # Revalidate
    for event in app.stream(None, thread, stream_mode="values"):
        pass
    
    final = app.get_state(thread)
    history = final.values.get('validation_history', [])
    
    print("\n📚 VALIDATION HISTORY")
    print("="*80)
    for i, entry in enumerate(history, 1):
        print(f"\nAttempt {i}:")
        print(f"  Timestamp: {entry['timestamp']}")
        print(f"  Valid: {entry['valid']}")
        if entry['errors']:
            print(f"  Errors:")
            for error in entry['errors']:
                print(f"    - {error}")
    print("="*80 + "\n")


# ============================================================================
# SOLUTION 5: Interactive Review Interface
# ============================================================================
def solution_5_review_interface():
    """Build interactive review and edit interface"""
    print("\n" + "="*80)
    print("SOLUTION 5: Interactive Review Interface")
    print("="*80 + "\n")
    
    workflow = StateGraph(AccountOpeningState)
    workflow.add_node("extract_data", extract_customer_data)
    workflow.add_node("validate_data", validate_customer_data)
    workflow.add_node("open_account", open_account)
    
    workflow.add_edge(START, "extract_data")
    workflow.add_edge("extract_data", "validate_data")
    workflow.add_edge("validate_data", "open_account")
    workflow.add_edge("open_account", END)
    
    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory,
        interrupt_after=["validate_data"]
    )
    
    thread = {"configurable": {"thread_id": "account_sol5"}}
    
    state = {
        "raw_application": "Lisa Chen wants checking account",
        "customer_name": "",
        "date_of_birth": "",
        "ssn_last_four": "",
        "address": "",
        "phone": "",
        "email": "",
        "initial_deposit": 0.0,
        "account_type": "",
        "data_validated": False,
        "validation_errors": [],
        "retry_count": 0,
        "validation_history": [],
        "age_verified": False,
        "deposit_limit_ok": False,
        "compliance_approved": False,
        "manager_approved": False,
        "manager_notes": "",
        "account_number": "",
        "final_status": "",
        "messages": []
    }
    
    for event in app.stream(state, thread, stream_mode="values"):
        pass
    
    current = app.get_state(thread)
    values = current.values
    
    # Display review interface
    print("\n" + "="*80)
    print("REVIEW INTERFACE - Edit Fields Below")
    print("="*80)
    
    fields_to_review = [
        ("customer_name", "Customer Name"),
        ("date_of_birth", "Date of Birth"),
        ("ssn_last_four", "SSN (Last 4)"),
        ("address", "Address"),
        ("phone", "Phone"),
        ("email", "Email"),
        ("initial_deposit", "Initial Deposit"),
        ("account_type", "Account Type")
    ]
    
    # Simulate interactive editing
    print("\n📝 Current Values:")
    for field, label in fields_to_review:
        value = values.get(field, "")
        status = "❌ NEEDS CORRECTION" if (not value or "NEEDS_REVIEW" in str(value) or value == 0.0) else "✅"
        print(f"{status} {label}: {value}")
    
    print("\n👤 Representative entering corrections...")
    
    # Simulated corrections
    corrections = {
        "customer_name": "Lisa Chen",
        "date_of_birth": "1992-08-30",
        "ssn_last_four": "5555",
        "address": "222 Market St, San Francisco, CA 94102",
        "phone": "415-555-2222",
        "email": "lisa.chen@email.com",
        "initial_deposit": 1000.0,
        "account_type": "checking",
        "data_validated": True,
        "validation_errors": []
    }
    
    app.update_state(thread, corrections)
    
    print("\n✅ All fields corrected and validated\n")
    
    for event in app.stream(None, thread, stream_mode="values"):
        pass


# ============================================================================
# SOLUTION 6: Add Compliance Checks
# ============================================================================
def check_compliance(state: AccountOpeningState) -> AccountOpeningState:
    """Check compliance requirements"""
    print("\n" + "="*80)
    print("📋 COMPLIANCE CHECK")
    print("="*80)
    
    # Check age (must be 18+)
    dob = state.get('date_of_birth', '')
    age_verified = False
    if dob and dob != "NEEDS_REVIEW":
        try:
            from datetime import datetime
            birth_date = datetime.strptime(dob, "%Y-%m-%d")
            age = (datetime.now() - birth_date).days / 365.25
            age_verified = age >= 18
            print(f"Age Check: {age:.0f} years old - {'✅ PASS' if age_verified else '❌ FAIL (Under 18)'}")
        except:
            print(f"Age Check: ❌ FAIL (Invalid date)")
    
    # Check deposit limits
    deposit = state.get('initial_deposit', 0)
    deposit_ok = 100 <= deposit <= 1000000
    print(f"Deposit Check: ${deposit:,.2f} - {'✅ PASS' if deposit_ok else '❌ FAIL'}")
    
    compliance_ok = age_verified and deposit_ok
    print(f"\nCompliance Status: {'✅ APPROVED' if compliance_ok else '❌ NEEDS REVIEW'}")
    print("="*80 + "\n")
    
    return {
        "age_verified": age_verified,
        "deposit_limit_ok": deposit_ok,
        "compliance_approved": compliance_ok
    }


def solution_6_compliance():
    """Add compliance checks"""
    print("\n" + "="*80)
    print("SOLUTION 6: Compliance Checks")
    print("="*80 + "\n")
    
    workflow = StateGraph(AccountOpeningState)
    workflow.add_node("extract_data", extract_customer_data)
    workflow.add_node("validate_data", validate_customer_data)
    workflow.add_node("check_compliance", check_compliance)
    workflow.add_node("open_account", open_account)
    
    workflow.add_edge(START, "extract_data")
    workflow.add_edge("extract_data", "validate_data")
    workflow.add_edge("validate_data", "check_compliance")
    workflow.add_edge("check_compliance", "open_account")
    workflow.add_edge("open_account", END)
    
    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory,
        interrupt_after=["check_compliance"]
    )
    
    # Test case 1: Underage applicant
    print("TEST 1: Underage Applicant")
    thread1 = {"configurable": {"thread_id": "compliance_1"}}
    state1 = {
        "raw_application": "Young person application",
        "customer_name": "Teen User",
        "date_of_birth": "2010-01-01",  # Too young
        "ssn_last_four": "1111",
        "address": "123 St",
        "phone": "555-0001",
        "email": "teen@email.com",
        "initial_deposit": 200.0,
        "account_type": "savings",
        "data_validated": True,
        "validation_errors": [],
        "retry_count": 0,
        "validation_history": [],
        "age_verified": False,
        "deposit_limit_ok": False,
        "compliance_approved": False,
        "manager_approved": False,
        "manager_notes": "",
        "account_number": "",
        "final_status": "",
        "messages": []
    }
    
    for event in app.stream(state1, thread1, stream_mode="values"):
        pass
    
    result1 = app.get_state(thread1)
    print(f"Result: Age Verified = {result1.values['age_verified']}\n")
    
    # Test case 2: Valid adult
    print("\nTEST 2: Valid Adult Applicant")
    thread2 = {"configurable": {"thread_id": "compliance_2"}}
    state2 = state1.copy()
    state2.update({
        "customer_name": "Adult User",
        "date_of_birth": "1985-01-01",
        "initial_deposit": 5000.0
    })
    
    for event in app.stream(state2, thread2, stream_mode="values"):
        pass
    
    result2 = app.get_state(thread2)
    print(f"Result: Compliance Approved = {result2.values['compliance_approved']}\n")


# ============================================================================
# SOLUTION 7: Test Incomplete Applications
# ============================================================================
def solution_7_incomplete_apps():
    """Test handling of incomplete applications"""
    print("\n" + "="*80)
    print("SOLUTION 7: Incomplete Application Handling")
    print("="*80 + "\n")
    
    workflow = StateGraph(AccountOpeningState)
    workflow.add_node("extract_data", extract_customer_data)
    workflow.add_node("validate_data", validate_customer_data)
    
    workflow.add_edge(START, "extract_data")
    workflow.add_edge("extract_data", "validate_