"""Text input with developer rules, context and option descriptions.

python examples/text_with_rules.py
"""

from smaug_decide import SmaugDecide


model = SmaugDecide.from_pretrained('abacusai/Smaug-Decisions')
message = 'My order arrived broken and nobody answers the phone. I want my money back now.'
print(
    model(
        message,
        'Which team should handle this?',
        ['billing', 'shipping', 'sales', 'abuse'],
        descriptions=['refunds and payments', 'delivery problems', None, 'threats or harassment'],
        rules='You route messages for an online shop. Refund requests go to billing even when the item was damaged in transit.',
    )
)
print(
    model(
        message,
        'How upset is the customer?',
        ['calm', 'annoyed', 'angry'],
        descriptions=['neutral tone', 'complains', 'threatens to leave'],
        rules='Judge the tone, not the facts.',
    )
)
