from pydantic import BaseModel, Field
from typing import Optional, List


class UserData(BaseModel):
    """Pydantic model for user profile data."""
    
    gender: int = Field(..., ge=0, le=1, description="0 (female) or 1 (male)")
    age: float = Field(..., ge=0.0, le=1.0, description="Normalized age [0, 1]")
    city: int = Field(..., ge=0, description="City ID")
    community_type: int = Field(..., ge=0, le=3, description="Community type (0-3)")
    city_level: int = Field(..., ge=1, le=6, description="City level (1-6)")
    price: float = Field(..., ge=0.0, le=1.0, description="Normalized price preference [0, 1]")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    history_pids: Optional[List[int]] = Field(None, description="List of integer PIDs from user's watch history (optional, for cold-start)")
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "gender": 1,
                "age": 0.54,
                "city": 225,
                "community_type": 2,
                "city_level": 2,
                "price": 0.046,
                "hour": 13,
                "day_of_week": 4,
                "history_pids": None
            }
        }
