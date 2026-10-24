"""
Step 6a: Realistic support-ticket text.

The first version of the generator used ~60 fixed sentences, so any classifier
scored 100% by memorising them. Real tickets are messy. Each ticket here is built
from parts, so the same (category, sentiment) can be written thousands of ways:

    [greeting] + [issue sentence for the category] + [detail] + [tone sentence] + [closing]

Realism added on purpose (this is what makes NLP hard in real life):
  * Hinglish tickets (~20%)          "mera order abhi tak nahi aaya"
  * typos / lowercase / !!!          "refnd not recieved"
  * polite complaints  -> negative   "I'm afraid the refund still hasn't arrived"
  * positive with a question         "love the product, how do I extend warranty?"
  * sarcasm -> negative              "Great, third delay this month. Amazing service."
  * cross-mentions                   a Billing ticket that also mentions refund
  * no tone at all (~35%)            "my refund has not come" -> negative or neutral? (truly ambiguous)
  * vague tickets (~8%)              "problem with my last order" -> category unclear

Only the `text` column is produced here; the labels (category, sentiment) are the
ones already decided in generate_data.py, so Steps 2-5 results do not change.
"""
import numpy as np

PRODUCTS = ["laptop", "office chair", "printer", "phone", "desk", "headphones", "monitor",
            "kurta", "shoes", "stapler", "paper ream", "sofa", "router", "jacket", "tablet"]

GREET = ["Hi,", "Hello team,", "Hi support,", "Dear team,", "Hey,", "", "", "", "Namaste,", "Hi there,"]
CLOSE = ["", "", "Thanks.", "Regards.", "Please help.", "Waiting for reply.", "Pls check asap.",
         "Thank you.", "Kindly do the needful."]

ISSUE = {
    "Delivery": {
        "en": ["my order #{oid} has not been delivered yet", "the {p} I ordered is delayed by {d} days",
               "the package arrived damaged", "delivery status shows delivered but I never got the {p}",
               "the courier did not show up on the promised date", "I got the wrong item instead of the {p}",
               "I need to change the delivery address for order #{oid}", "where is my shipment, tracking is not updating",
               "can you share the tracking link for my {p}", "half of my order is missing, only got the {p}"],
        "hi": ["mera order #{oid} abhi tak nahi aaya", "{p} ki delivery {d} din se late hai",
               "parcel toota hua aaya hai", "tracking update nahi ho rahi", "galat item bhej diya, {p} mangaya tha",
               "delivery boy aaya hi nahi"],
    },
    "Billing": {
        "en": ["I was charged twice for order #{oid}", "the invoice amount is wrong, it shows Rs {amt}",
               "please send the GST invoice for my {p}", "my card was debited but the order failed",
               "the discount was not applied at checkout", "how can I update my billing address",
               "I see an extra charge of Rs {amt} on my statement", "EMI option is not showing on my card",
               "the coupon code gave an error during payment"],
        "hi": ["paise do baar kat gaye", "invoice mein amount galat hai, Rs {amt} dikha raha hai",
               "GST bill chahiye {p} ka", "payment ho gaya par order fail ho gaya", "discount apply nahi hua"],
    },
    "Product Quality": {
        "en": ["the {p} stopped working within a week", "the {p} is not as described on the website",
               "the stitching on the {p} came off after one wash", "the {p} makes a strange noise",
               "the colour of the {p} is completely different", "is there a warranty on the {p}",
               "the {p} has scratches out of the box", "battery of the {p} drains very fast",
               "what material is the {p} made of"],
        "hi": ["{p} ek hafte mein kharab ho gaya", "{p} website jaisa bilkul nahi hai", "{p} ki quality bekar hai",
               "{p} pe warranty hai kya", "{p} mein scratch hai"],
    },
    "Technical": {
        "en": ["the app crashes when I try to checkout", "I cannot log in to my account",
               "OTP is not coming on my phone", "the website keeps showing an error 500",
               "how do I reset my password", "the payment page is stuck on loading",
               "I cannot download my order history", "notifications are not working after the update",
               "the search on the app shows no results"],
        "hi": ["app bar bar crash ho raha hai", "login nahi ho raha", "OTP nahi aa raha",
               "password reset kaise karu", "payment page atak gaya hai"],
    },
    "Refund": {
        "en": ["my refund for order #{oid} has not been received", "it has been {d} days and still no refund",
               "the refund amount is less than what I paid", "I returned the {p} but the refund is pending",
               "what is the refund timeline for returned items", "I want to cancel order #{oid} and get a refund",
               "the refund went to the wrong account"],
        "hi": ["refund abhi tak nahi aaya", "{d} din ho gaye refund nahi mila", "{p} return kiya par paise nahi aaye",
               "refund kitne din mein aayega", "order cancel karke refund chahiye"],
    },
}

VAGUE = {  # same wording used for every category -> category can't be read from the text
    "en": ["I have an issue with order #{oid}", "there is a problem with my last purchase",
           "need help with my recent order", "something is wrong with my {p} order",
           "please call me regarding order #{oid}"],
    "hi": ["mere order #{oid} mein problem hai", "last order ke baare mein help chahiye"],
}

CROSS = {  # a second, smaller mention of another topic (makes category harder)
    "Billing": ["also the refund is still pending", "and the delivery was late too"],
    "Refund": ["the invoice was also wrong", "also I was charged an extra amount"],
    "Delivery": ["and the invoice is missing", "also the product looks used"],
    "Product Quality": ["delivery was fine though", "I want a refund or replacement"],
    "Technical": ["because of this my payment failed", "so I could not track my delivery"],
}

TONE = {
    "negative": {
        "en": ["This is really frustrating.", "Very disappointed with your service.", "Worst experience ever.",
               "This is unacceptable.", "I am fed up of following up.", "Third time I am complaining!!",
               "If not fixed I will cancel my account.", "Totally useless support.",
               "I will never order again.", "Nobody is responding to my emails."],
        "polite": ["I'm afraid this is still not resolved.", "Unfortunately this has happened again.",
                   "I would appreciate it if this is sorted soon, it has been a while.",
                   "Sadly I am not happy with this."],
        "sarcasm": ["Great, another delay. Amazing service.", "Wow, what a fantastic experience. Not.",
                    "Thanks a lot for wasting my week."],
        "hi": ["Bahut bura experience hai.", "Ye bilkul acceptable nahi hai.", "Kitni baar complain karu?",
               "Bakwas service.", "Paisa barbaad ho gaya."],
    },
    "neutral": {
        "en": ["Please check.", "Can you help with this?", "Let me know the next steps.",
               "Please share an update.", "What should I do?", ""],
        "hi": ["Please check karo.", "Kya karna hoga batao.", "Update dena please.", ""],
    },
    "positive": {
        "en": ["Thanks for the quick support last time!", "Love the product otherwise.",
               "Great service as always, just a small query.", "You guys are awesome.",
               "Really happy with my purchase.", "Appreciate the fast response."],
        "hi": ["Product bahut accha hai.", "Service mast hai, bas ek sawal tha.", "Thank you, aap log best ho."],
    },
}


def _typo(text: str, rng) -> str:
    chars = list(text)
    for _ in range(rng.integers(1, 4)):
        i = int(rng.integers(1, max(2, len(chars) - 1)))
        op = rng.integers(0, 3)
        if op == 0 and chars[i].isalpha():
            del chars[i]                                  # drop a letter
        elif op == 1 and i + 1 < len(chars):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]  # swap
        else:
            chars.insert(i, chars[i])                      # double a letter
    return "".join(chars)


def ticket_text(category: str, sentiment: str, rng) -> str:
    hinglish = rng.random() < 0.20
    lang = "hi" if hinglish else "en"
    slots = {"oid": int(rng.integers(100000, 999999)), "d": int(rng.integers(3, 25)),
             "amt": int(rng.integers(199, 49999)), "p": rng.choice(PRODUCTS)}
    pool = VAGUE[lang] if rng.random() < 0.08 else ISSUE[category][lang]
    issue = rng.choice(pool).format(**slots)

    if rng.random() < 0.35:
        tone = ""                     # many real tickets state the issue with no emotion
    elif sentiment == "negative":
        r = rng.random()
        if hinglish:
            tone = rng.choice(TONE["negative"]["hi"])
        elif r < 0.15:
            tone = rng.choice(TONE["negative"]["polite"])
        elif r < 0.22:
            tone = rng.choice(TONE["negative"]["sarcasm"])
        else:
            tone = rng.choice(TONE["negative"]["en"])
    else:
        tone = rng.choice(TONE[sentiment][lang])

    parts = [rng.choice(GREET), issue[0].upper() + issue[1:] + "."]
    if rng.random() < 0.12:
        parts.append(rng.choice(CROSS[category]).capitalize() + ".")
    # positive tickets usually lead with the praise, then the question
    if tone:
        parts = parts[:1] + [tone] + parts[1:] if sentiment == "positive" and rng.random() < 0.6 else parts + [tone]
    parts.append(rng.choice(CLOSE))
    text = " ".join(p for p in parts if p).strip()

    if rng.random() < 0.15:
        text = _typo(text, rng)
    if rng.random() < 0.15:
        text = text.lower()
    if sentiment == "negative" and rng.random() < 0.15:
        text = text.rstrip(".") + "!!!"
    return text


def make_ticket_texts(tickets, rng) -> list:
    return [ticket_text(c, s, rng) for c, s in zip(tickets["category"], tickets["sentiment"])]
