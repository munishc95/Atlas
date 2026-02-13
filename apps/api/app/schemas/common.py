from pydantic import BaseModel


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict | list | str | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody
