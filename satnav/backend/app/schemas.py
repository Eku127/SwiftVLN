"""Request/response schemas for SatNav API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class BackendLoginRequest(BaseModel):
    """Credentials for CloudSDK login when no cached x-auth-token exists."""

    username: Optional[str] = None
    password: Optional[str] = None
    flag: int = Field(default=1, ge=1)


class ModelInferenceRequest(BaseModel):
    """Run one deploy image step after capturing a fresh RTMP frame."""

    instruction: str = Field(min_length=1)


class FlightRcBackendDrcRequest(BaseModel):
    """Acquire DRC control via serial connect + enter backend calls."""

    expire_sec: int = Field(
        default=3600,
        ge=1800,
        le=86400,
        description="DRC session validity in seconds",
    )


class FlightRcBackendDeviceRegisterRequest(BaseModel):
    """Register RC controller SN and aircraft device SN for flight control."""

    rc_sn: str = Field(min_length=1, description="RC Plus 2 controller serial number")
    device_sn: str = Field(min_length=1, description="Aircraft device serial number")


class FlightForwardRequest(BaseModel):
    """Submit a forward stick task (action=FORWARD)."""

    distance_m: Optional[float] = Field(default=None, description="Forward distance in meters")
    tolerance_m: Optional[float] = Field(default=None, ge=0, le=50)
    timeout_ms: Optional[int] = Field(default=None, ge=1000, le=300000)


class FlightTurnRequest(BaseModel):
    """Submit a yaw stick task (action=TURN_LEFT or TURN_RIGHT)."""

    action: int = Field(ge=2, le=3, description="2=TURN_LEFT, 3=TURN_RIGHT")
    degree: Optional[float] = Field(
        default=None,
        gt=0,
        description="Absolute yaw angle in degrees; sign comes from action",
    )
    tolerance_deg: Optional[float] = Field(default=None, ge=0, le=10)
    timeout_ms: Optional[int] = Field(default=None, ge=1000, le=300000)
