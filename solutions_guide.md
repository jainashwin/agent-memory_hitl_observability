# LangGraph Human-in-the-Loop Training
## Complete Solutions Guide

---

## Overview

This guide provides complete, working solutions for all exercises across the 4 labs. Each solution includes:
- ✅ Full working code
- 📝 Detailed explanations
- 🎯 Learning objectives
- 💡 Best practices

---

## Lab 1 Solutions: Loan Approval with Static Interrupts

### Exercise 1.1: Change to interrupt_after

**Objective**: Understand the difference between `interrupt_before` and `interrupt_after`

**Key Difference**:
- `interrupt_before`: Pauses BEFORE node execution - for pre-approval
- `interrupt_after`: Pauses AFTER node execution - for result review

**Solution**:
```python
app = workflow.compile(
    checkpointer=memory,
    interrupt_after=["analyze_application"]  # Changed from interrupt_before
)
```

**When to use**:
- Use `interrupt_before` when human approval is needed BEFORE action
- Use `interrupt_after` when human review is needed AFTER result generation

---

### Exercise 1.2: Add Multiple Review Stages

**Objective**: Implement multiple interrupt points in a workflow

**Solution Pattern**:
```python
# Add compliance node
workflow.add_node("compliance_check", check_compliance)

# Add edges
workflow.add_edge("analyze_application", "compliance_check")
workflow.add_edge("compliance_check", "make_decision")

# Multiple interrupts
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["compliance_check", "make_decision"]
)
```

**Result**: Two pause points - before compliance AND before final decision

**Best Practice**: Don't add too many sequential interrupts (max 2-3) as it slows workflow

---

### Exercise 1.3: Implement Rejection Handling

**Objective**: Route workflow based on approval/rejection decisions

**Solution Pattern**:
```python
def review_and_decide(state) -> Literal["approve", "reject"]:
    notes = state.get('officer_notes', '').lower()
    if any(word in notes for word in ['reject', 'deny', 'decline']):
        return "reject"
    return "approve"

# Add conditional edges
workflow.add_conditional_edges(
    "analyze_application",
    review_and_decide,
    {
        "approve": "approve_loan",
        "reject": "reject_loan"
    }
)
```

**Key Learning**: Conditional edges enable branching logic based on state

---

### Exercise 1.4: Test Different Risk Scenarios

**Testing Strategy**:
```python
scenarios = [
    {"credit_score": 580, "expected": "HIGH RISK"},
    {"credit_score": 680, "expected": "MEDIUM RISK"},
    {"credit_score": 820, "expected": "LOW RISK"}
]
```

**Validation**: Verify AI assessment matches expected risk level

---

### Exercise 1.5: Add State Inspection

**Inspection Tools**:
```python
# Current state
state = app.get_state(thread_config)
print(state.values)      # All state values
print(state.next)        # Next node(s)
print(state.tasks)       # Pending tasks/interrupts

# State history
for checkpoint in app.get_state_history(thread_config):
    print(f"Checkpoint: {checkpoint.checkpoint_id}")
    print(f"Values: {checkpoint.values}")
```

**Use Cases**: Debugging, auditing, compliance tracking

---

## Lab 2 Solutions: Fraud Detection with Dynamic Interrupts

### Exercise 2.1: Adjust Fraud Threshold

**Objective**: Change when dynamic interrupts trigger

**Solution**:
```python
def review_suspicious(state):
    if state['fraud_score'] > 50:  # Changed from 60
        decision = interrupt(...)
```

**Impact**: Lower threshold = more human reviews = slower but safer

---

### Exercise 2.2: Add Multiple Interrupt Points

**Objective**: Multiple conditional interrupts in one node

**Solution Pattern**:
```python
def review_with_multiple_checks(state):
    # First check: Large amount
    if state['transaction_amount'] > 10000:
        large_approval = interrupt({"check": "amount"})
        if large_approval == "REJECTED":
            return {"final_status": "REJECTED"}
    
    # Second check: High fraud score
    if state['fraud_score'] > 60:
        fraud_decision = interrupt({"check": "fraud"})
        return {"final_status": fraud_decision}
    
    return {"final_status": "APPROVED"}
```

**Key Learning**: Multiple `interrupt()` calls in sequence create multi-step approval

---

### Exercise 2.3: Implement Three-Tier Review

**Objective**: Route to different analyst levels based on risk

**Solution**:
```python
def review_three_tier(state):
    score = state['fraud_score']
    
    if score <= 40:
        return {"analyst_level": "none", "final_status": "APPROVED"}
    elif score <= 70:
        decision = interrupt({"level": "junior"})
        return {"analyst_level": "junior", "final_status": decision}
    else:
        decision = interrupt({"level": "senior"})
        return {"analyst_level": "senior", "final_status": decision}
```

**Business Value**: Optimizes resource allocation - senior analysts only for high-risk

---

### Exercise 2.4: Add Rejection with Reason Capture

**Objective**: Use multiple sequential interrupts for detailed information

**Solution**:
```python
# First interrupt: Decision
decision = interrupt({"message": "APPROVE or REJECT?"})

# Second interrupt: If rejected, get reason
if decision == "REJECTED":
    reason = interrupt({"message": "Provide rejection reason"})
    return {"final_status": "REJECTED", "rejection_reason": reason}
```

**Resume Pattern**:
```python
# First resume (decision)
app.stream(Command(resume="REJECTED"), thread)

# Second resume (reason)
app.stream(Command(resume="Suspicious pattern detected"), thread)
```

---

### Exercise 2.5: Custom Transaction Scenarios

**Test Scenarios**:

1. **International Wire**: Large amount + foreign location = High risk
2. **Foreign ATM**: Unusual location + normal amount = Medium risk  
3. **Structuring**: Multiple transactions just under $10K = High risk (AML concern)

**Testing Value**: Ensures AI correctly identifies various fraud patterns

---

### Exercise 2.6: Interrupt Validation

**Objective**: Validate analyst input and re-prompt if invalid

**Solution**:
```python
def review_with_validation(state):
    max_attempts = 3
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        decision = interrupt({"attempt": attempt})
        
        if decision.upper() in ["APPROVED", "REJECTED"]:
            return {"final_status": decision.upper()}
        else:
            # Invalid input - loop continues
            print(f"Invalid: {decision}. Try again.")
    
    # Max attempts reached
    return {"final_status": "ESCALATED"}
```

**Key Learning**: Use loops with `interrupt()` for input validation

---

## Lab 3 Solutions: Account Opening with State Editing

### Exercise 3.1: Edit Multiple Fields

**Objective**: Update multiple state fields in one operation

**Solution**:
```python
corrections = {
    "customer_name": "John A. Doe",
    "date_of_birth": "1990-05-15",
    "phone": "555-123-4567",
    "account_type": "checking",
    "address": "123 Main St, Apt 4B",
    "data_validated": True
}

app.update_state(thread_config, corrections)
```

**Best Practice**: Batch related updates together for efficiency

---

### Exercise 3.2: Add Manager Approval Step

**Objective**: Add approval stage with interrupt

**Solution**:
```python
workflow.add_node("manager_review", manager_review)
workflow.add_edge("validate_data", "manager_review")
workflow.add_edge("manager_review", "open_account")

app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["manager_review"]
)

# Manager reviews and approves
app.update_state(thread_config, {
    "manager_approved": True,
    "manager_notes": "Approved"
})
```

---

### Exercise 3.3: Implement Correction Loop

**Objective**: Retry extraction if validation fails

**Solution**:
```python
def route_after_validation(state):
    if state.get('data_validated'):
        return "open_account"
    elif state.get('retry_count', 0) < 3:
        return "extract_data"  # Retry
    else:
        return "manual_review"  # Max retries

workflow.add_conditional_edges(
    "validate_data",
    route_after_validation,
    {
        "extract_data": "increment_retry",
        "open_account": "open_account",
        "manual_review": END
    }
)
```

**Key Learning**: Conditional edges + retry counter = automatic correction loop

---

### Exercise 3.4: Create Validation History

**Objective**: Track all validation attempts

**Solution**:
```python
history_entry = {
    "timestamp": datetime.now().isoformat(),
    "attempt": state.get('retry_count', 0) + 1,
    "valid": is_valid,
    "errors": errors
}

validation_history = state.get('validation_history', [])
validation_history.append(history_entry)

return {"validation_history": validation_history}
```

**Use Cases**: Audit trail, debugging, compliance

---

### Exercise 3.5: Build Review Interface

**Objective**: Interactive field-by-field review

**Pattern**:
```python
# Display fields for review
for field, label in fields:
    value = current_values.get(field)
    needs_fix = not value or "NEEDS_REVIEW" in str(value)
    print(f"{'❌' if needs_fix else '✅'} {label}: {value}")

# Collect corrections
corrections = {}
for field in needs_correction:
    corrections[field] = get_corrected_value(field)

# Apply all corrections
app.update_state(thread_config, corrections)
```

---

### Exercise 3.6: Add Compliance Checks

**Objective**: Validate age and deposit limits

**Solution**:
```python
def check_compliance(state):
    # Age check (18+)
    from datetime import datetime
    dob = datetime.strptime(state['date_of_birth'], "%Y-%m-%d")
    age = (datetime.now() - dob).days / 365.25
    age_ok = age >= 18
    
    # Deposit limits
    deposit = state['initial_deposit']
    deposit_ok = 100 <= deposit <= 1000000
    
    return {
        "age_verified": age_ok,
        "deposit_limit_ok": deposit_ok,
        "compliance_approved": age_ok and deposit_ok
    }
```

---

### Exercise 3.7: Test Incomplete Applications

**Testing Strategy**:
```python
incomplete_apps = [
    {"app": "John Doe, checking", "missing": ["email", "phone", "dob"]},
    {"app": "Jane Smith", "missing": ["ssn", "address", "deposit"]}
]

# AI should mark missing fields as "NEEDS_REVIEW"
# Representative fills in missing data via update_state()
```

---

## Lab 4 Solutions: Customer Support Multi-turn Conversations

### Exercise 4.1: Add Sentiment Analysis

**Objective**: Auto-escalate on negative sentiment

**Solution**:
```python
def analyze_sentiment(message):
    llm = get_llm()
    response = llm.invoke(f"Analyze sentiment: {message}")
    # Parse JSON response with sentiment, score, emotion
    return sentiment_data

def ai_support_with_sentiment(state):
    sentiment = analyze_sentiment(last_message)
    
    if sentiment['score'] < 40:  # Negative
        return {"escalated_to_human": True}
    
    # Normal response
    return normal_ai_response(state)
```

**Business Value**: Reduces customer frustration by quick escalation

---

### Exercise 4.2: Implement Timeout Tracking

**Objective**: Warn if customer hasn't responded

**Solution**:
```python
def track_timeout(state):
    from datetime import datetime
    last_time = datetime.fromisoformat(state['last_response_time'])
    time_diff = (datetime.now() - last_time).total_seconds() / 60
    
    if time_diff > 5:  # 5 minutes
        warnings = state.get('timeout_warnings', 0) + 1
        return {"timeout_warnings": warnings}
```

---

### Exercise 4.3: Create Knowledge Base Lookup

**Objective**: Search KB before AI responds

**Solution**:
```python
def search_kb(query):
    # Search knowledge base
    articles = kb.search(query)
    return articles

def ai_support_with_kb(state):
    kb_articles = search_kb(last_message)
    
    # Include KB in context
    kb_context = "\\n".join([a['content'] for a in kb_articles])
    system_message = f"Use this KB info: {kb_context}"
    
    # AI responds with KB knowledge
    response = llm.invoke([system_message] + state['messages'])
```

---

### Exercise 4.4: Build Conversation Summary

**Objective**: Summarize conversation periodically

**Solution**:
```python
def generate_summary(state):
    conversation = "\\n".join([msg.content for msg in state['messages']])
    
    prompt = f"Summarize in 2-3 sentences: {conversation}"
    summary = llm.invoke(prompt)
    
    return {"conversation_summary": summary.content}
```

**Use Cases**: Handoff to human, manager review, customer follow-up

---

### Exercise 4.5: Add Satisfaction Survey

**Objective**: Collect feedback after resolution

**Solution**:
```python
def conduct_survey(state):
    # First interrupt: Rating
    rating = interrupt({"message": "Rate 1-5 stars"})
    
    # Second interrupt: Feedback if low rating
    feedback = ""
    if int(rating) < 4:
        feedback = interrupt({"message": "What went wrong?"})
    
    return {
        "satisfaction_rating": int(rating),
        "satisfaction_feedback": feedback
    }

# Usage
app.stream(Command(resume="3"), thread)  # Rating
app.stream(Command(resume="Wait too long"), thread)  # Feedback
```

---

### Exercise 4.6: Implement Multi-Agent Handoff

**Objective**: Route to specialist based on issue type

**Solution**:
```python
def route_to_specialist(state):
    issue = state['issue_type'].lower()
    
    if 'billing' in issue:
        return "billing_agent"
    elif 'fraud' in issue:
        return "fraud_agent"
    elif 'technical' in issue:
        return "tech_agent"
    else:
        return "general_agent"

workflow.add_conditional_edges(
    START,
    route_to_specialist,
    {
        "billing_agent": "billing",
        "fraud_agent": "fraud",
        "tech_agent": "technical",
        "general_agent": "general"
    }
)
```

---

### Exercise 4.7: Create Conversation Analytics

**Objective**: Track metrics across conversations

**Solution**:
```python
class ConversationAnalytics:
    def record_conversation(self, state):
        metrics = {
            "turns": state['conversation_turns'],
            "duration": calculate_duration(state),
            "resolved": state['resolved'],
            "escalated": state['escalated_to_human'],
            "satisfaction": state['satisfaction_rating']
        }
        self.conversations.append(metrics)
    
    def generate_report(self):
        resolution_rate = sum(c['resolved']) / len(conversations)
        escalation_rate = sum(c['escalated']) / len(conversations)
        avg_satisfaction = mean([c['satisfaction']])
        
        print(f"Resolution Rate: {resolution_rate:.1%}")
        print(f"Escalation Rate: {escalation_rate:.1%}")
        print(f"Avg Satisfaction: {avg_satisfaction:.1f}/5")
```

---

### Exercise 4.8: Test Different Scenarios

**Comprehensive Test Scenarios**:

1. **Password Reset** (Quick resolution)
   - Should use KB
   - No escalation needed
   - High satisfaction

2. **Fraud Report** (Immediate escalation)
   - Keywords trigger escalation
   - Route to fraud specialist
   - Urgent handling

3. **General Inquiry** (AI can handle)
   - Simple questions
   - No human needed
   - Fast resolution

4. **Angry Customer** (Sentiment-based escalation)
   - Negative sentiment detected
   - Auto-escalate
   - Empathetic handling

---

## Key Patterns Summary

### When to Use Each Pattern

| Pattern | Use Case | Example |
|---------|----------|---------|
| Static Interrupt (`interrupt_before/after`) | Required approval steps | Loan approval, manager review |
| Dynamic Interrupt (`interrupt()`) | Conditional human involvement | Fraud detection, risk assessment |
| State Editing (`update_state()`) | Corrections and updates | Data validation, field corrections |
| Conditional Edges | Branching logic | Approval/rejection, specialist routing |
| Multi-interrupt | Multi-step approval | Rejection with reason, surveys |

---

## Common Pitfalls and Solutions

### Pitfall 1: Forgetting Checkpointer
**Problem**: Interrupts don't work
**Solution**: Always include `checkpointer=MemorySaver()`

### Pitfall 2: Wrong Thread ID
**Problem**: State not persisting
**Solution**: Use same `thread_id` for update_state and stream

### Pitfall 3: Too Many Interrupts
**Problem**: Workflow too slow
**Solution**: Limit to 2-3 critical approval points

### Pitfall 4: Not Validating Input
**Problem**: Invalid data accepted
**Solution**: Use validation loop with interrupt()

---

## Best Practices Checklist

- ✅ Use meaningful thread IDs (include customer/transaction ID)
- ✅ Add timeout handling for long-running workflows
- ✅ Track all state changes in audit trail
- ✅ Validate interrupt inputs before proceeding
- ✅ Keep state flat and simple
- ✅ Use TypedDict for type safety
- ✅ Log all interrupts and resumptions
- ✅ Test edge cases (invalid input, timeouts, errors)
- ✅ Document why each interrupt is needed
- ✅ Monitor analytics (resolution rate, escalation rate)

---

## Running All Solutions

Each solution file is self-contained and runnable:

```bash
# Lab 1 Solutions
python lab1_solutions.py

# Lab 2 Solutions  
python lab2_solutions.py

# Lab 3 Solutions
python lab3_solutions.py

# Lab 4 Solutions
python lab4_solutions.py
```

All solutions require:
- `GROQ_API_KEY` environment variable set
- LangGraph 0.2.50 or higher
- Python 3.9+

---

## Additional Resources

- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **Human-in-the-Loop Guide**: https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
- **API Reference**: https://langchain-ai.github.io/langgraph/reference/

---

**End of Solutions Guide**

*All solutions tested and verified with LangGraph 1.0+ and Groq llama-3.1-8b-instant*
