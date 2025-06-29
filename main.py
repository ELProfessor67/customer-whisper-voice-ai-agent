"""Webhook server to handle Twilio calls and start the voice bot."""
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from utils.daily_helpers import create_sip_room
from bot import run_bot
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import aiohttp
from fastapi.responses import JSONResponse
# Load environment variables
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create aiohttp session to be used for Daily API calls
    app.state.session = aiohttp.ClientSession()
    yield
    # Close session when shutting down
    await app.state.session.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/start", response_class=PlainTextResponse)
async def handle_call(request: Request):
    """Handle incoming Twilio call webhook."""
    print("Received call webhook from Twilio")

    try:
        # Get form data from Twilio webhook
        form_data = await request.form()
        data = dict(form_data)

        # Extract call ID (required to forward the call later)
        call_sid = data.get("CallSid")
        if not call_sid:
            raise HTTPException(status_code=400, detail="Missing CallSid in request")

        # Extract the caller's phone number
        caller_phone = str(data.get("From", "unknown-caller"))
        print(f"Processing call with ID: {call_sid} from {caller_phone}")

        # Create a Daily room with SIP capabilities
        try:
            room_details = await create_sip_room(request.app.state.session, caller_phone)
        except Exception as e:
            print(f"Error creating Daily room: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create Daily room: {str(e)}")

        # Extract necessary details
        room_url = room_details["room_url"]
        token = room_details["token"]
        sip_endpoint = room_details["sip_endpoint"]

        # Make sure we have a SIP endpoint
        if not sip_endpoint:
            raise HTTPException(status_code=500, detail="No SIP endpoint provided by Daily")

        try:
            
            asyncio.create_task(run_bot(room_url, token, call_sid, sip_endpoint))
            print(f"Started async bot for call: {call_sid}")

        except Exception as e:
            print(f"Error starting bot: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start bot: {str(e)}")


        return JSONResponse(
            content={
                "token": token,
                "call_sid": call_sid,
                "sip_endpoint": sip_endpoint,
                "room_url": room_url
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}
