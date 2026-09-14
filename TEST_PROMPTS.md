# Test Prompts

Use these against the sample document (`sample_docs/acme_handbook.txt`) to verify the chatbot end to end. Type only the question text into the chat box — not the arrows or expected-answer notes below, those are just for your reference.

## Set 1 — 20 Core Test Questions

**Direct facts (should answer correctly from the document)**
1. How many paid vacation days do full-time employees get per year?
2. How many paid sick days do employees get?
3. Which days of the week must employees work from the office?
4. Can I work remotely on Wednesdays?
5. What's required for a fully remote work arrangement?
6. How far in advance must I submit a vacation request?
7. How many unused vacation days can carry over to next year?
8. What expenses can be reimbursed?
9. How long do I have to submit a reimbursement receipt?
10. How many business days does reimbursement processing take?
11. Is my home internet bill reimbursable?
12. Are gym memberships covered?
13. What do new employees receive in their first 90 days?
14. What happens to company equipment when I leave?

**Slightly indirect / requires synthesis**
15. If I want to work from home every day, what do I need to do?
16. Summarize the leave policy in one sentence.
17. What benefits does a new hire get in their first three months?

**Out-of-scope (should refuse, not hallucinate)**
18. Does the company offer health insurance or a 401k?
19. What is the capital of France?

**Conversation memory (send as two separate messages)**
20. First: "My name is Alex and I work in engineering." Then: "What's my name and what team am I on?"

## Set 2 — Hybrid Grounding Demo (document facts vs. general knowledge/math)

These show the chatbot correctly separating "must come from the document" facts from general knowledge/math, which it will answer directly instead of refusing.

**Document-grounded (strict)**
1. "How many paid vacation days do full-time employees get?" -- expected: 18 days, from the document.
2. "Does Acme Robotics offer a 401k match?" -- expected: "I don't have that information" (correctly refuses instead of guessing).

**General knowledge / math (answered freely, not blocked by the document)**
3. "What is 347 x 12?"
4. "What's the difference between RAM and ROM?"
5. "Convert 100 F to Celsius."

**Mixed in one turn (memory + math together)**
6. First: "My name is Alex." -- bot remembers it.
7. Then: "What's my name, and what's 15% of 200?" -- expected: answers both in one reply, one from conversation memory, one from math -- while a pure document question in the same turn would still pull from the handbook instead.
