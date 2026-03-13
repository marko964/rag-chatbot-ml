"""Cal.com v2 API client."""
import httpx
from app.config import settings

_HEADERS = {
    "Content-Type": "application/json",
    "cal-api-version": "2024-08-13",
}


def _auth_headers() -> dict:
    return {**_HEADERS, "Authorization": f"Bearer {settings.calcom_api_key}"}


async def get_available_slots(date: str) -> dict:
    """
    Get available slots for a date (YYYY-MM-DD).
    Returns the raw Cal.com response or an error dict.
    """
    if not settings.calcom_api_key or not settings.calcom_event_type_id:
        return {"error": "Cal.com is not configured on this server."}

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(
                f"{settings.calcom_base_url}/slots/available",
                params={
                    "eventTypeId": settings.calcom_event_type_id,
                    "startTime": f"{date}T00:00:00Z",
                    "endTime": f"{date}T23:59:59Z",
                },
                headers=_auth_headers(),
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            return {"error": f"Failed to fetch slots: {str(e)}"}


async def create_booking(
    name: str,
    email: str,
    start_time: str,
    notes: str = "",
) -> dict:
    """
    Book a meeting. start_time must be ISO 8601 (e.g. 2024-06-15T14:00:00Z).
    Returns {"success": bool, "data": ...}.
    """
    if not settings.calcom_api_key or not settings.calcom_event_type_id:
        return {"success": False, "error": "Cal.com is not configured on this server."}

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.post(
                f"{settings.calcom_base_url}/bookings",
                json={
                    "eventTypeId": int(settings.calcom_event_type_id),
                    "start": start_time,
                    "attendee": {
                        "name": name,
                        "email": email,
                        "timeZone": "UTC",
                    },
                    "metadata": {"notes": notes} if notes else {},
                },
                headers=_auth_headers(),
            )
            resp.raise_for_status()
            return {"success": True, "data": resp.json()}
        except httpx.HTTPStatusError as e:
            return {"success": False, "error": e.response.text}
        except httpx.HTTPError as e:
            return {"success": False, "error": str(e)}
