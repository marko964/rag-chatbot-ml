from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional

from app.db.database import get_db
from app.db.models import Lead
from app.config import settings

router = APIRouter()


def verify_admin(x_api_key: str = Header(...)):
    if x_api_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key.")


class LeadOut(BaseModel):
    id: int
    name: str
    email: str
    phone: Optional[str]
    company: Optional[str]
    notes: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


@router.get("/admin/leads", response_model=list[LeadOut], dependencies=[Depends(verify_admin)])
def list_leads(db: Session = Depends(get_db)):
    return db.query(Lead).order_by(Lead.created_at.desc()).all()
