"""Razorpay payments router."""

import razorpay
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from medscan.config import get_settings

router = APIRouter()

class OrderRequest(BaseModel):
    amount_inr: int
    plan_name: str

class VerifyRequest(BaseModel):
    razorpay_payment_id: str
    razorpay_order_id: str
    razorpay_signature: str

def get_razorpay_client():
    settings = get_settings()
    if not settings.razorpay_key_id or not settings.razorpay_key_secret:
        raise HTTPException(status_code=500, detail="Razorpay keys not configured")
    return razorpay.Client(auth=(settings.razorpay_key_id, settings.razorpay_key_secret))

@router.post("/create-order")
async def create_order(req: OrderRequest):
    """Create a new Razorpay order."""
    client = get_razorpay_client()
    try:
        # Razorpay expects amount in paise (1 INR = 100 paise)
        order = client.order.create({
            "amount": req.amount_inr * 100,
            "currency": "INR",
            "receipt": f"receipt_{req.plan_name.lower().replace(' ', '_')}"
        })
        return {"order_id": order["id"], "amount": order["amount"], "currency": order["currency"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/verify")
async def verify_payment(req: VerifyRequest):
    """Verify Razorpay payment signature."""
    client = get_razorpay_client()
    try:
        client.utility.verify_payment_signature({
            'razorpay_order_id': req.razorpay_order_id,
            'razorpay_payment_id': req.razorpay_payment_id,
            'razorpay_signature': req.razorpay_signature
        })
        return {"status": "success", "message": "Payment verified successfully"}
    except razorpay.errors.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid payment signature")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
