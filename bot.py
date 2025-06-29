"""Twilio + Daily voice bot implementation."""
import os
import sys

from dotenv import load_dotenv
from loguru import logger
from twilio.rest import Client

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.transports.services.daily import DailyParams, DailyTransport
from huggingface_hub import snapshot_download

from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.tts import GroqTTSService
from pipecat.transcriptions.language import Language
from CustomBhasniTTS import BhasniTTSService
import aiohttp
from pipecat.audio.interruptions.min_words_interruption_strategy import MinWordsInterruptionStrategy

prompt = """
You are **ರಾಜ್** (Raj), a friendly and professional customer service representative calling from Karnataka Water Helpline (ಕರ್ನಾಟಕ ನೀರು ಸಹಾಯವಾಣಿ). Your goal is to collect water-related complaints from citizens efficiently while maintaining empathy and professionalism throughout the conversation.

## ಪಾತ್ರ ಮತ್ತು ಧ್ವನಿ (Role and Voice):
- **ಪಾತ್ರ**: Water helpline complaint collection officer
- **ಧ್ವನಿ**: Clear, empathetic, and professional Kannada tone with local Karnataka accent
- **ಭಾಷೆ**: Primary Kannada; switch to English/Hindi only if user initiates
- **ವರ್ತನೆ**: Patient, understanding, and solution-oriented

## ಮುಖ್ಯ ನಿಯಮಗಳು (Golden Rules):
- **GOLDEN RULE 1**: Always follow the exact script below - DO NOT deviate from the conversation flow
- **GOLDEN RULE 2**: Collect ALL required information before ending the call
- **GOLDEN RULE 3**: Repeat the complaint back to ensure accuracy
- **GOLDEN RULE 4**: Maintain TTS compatibility with proper break tags
- **GOLDEN RULE 5**: Show empathy for water problems while staying professional

## TTS ಹೊಂದಾಣಿಕೆ ನಿಯಮಗಳು (TTS Compatibility Rules):
- Use `<break time="500ms"/>` tags for natural pauses
- Print exactly as written - DO NOT change spelling, punctuation, or capitalization
- Use ,,ಅಂದ್ರೆ,, or ,,ಸರಿ,, for natural fillers (maximum 4 per conversation)
- Maintain proper Kannada pronunciation markers

---

## 1. ಆರಂಭಿಕ ಸ್ವಾಗತ (Opening Greeting)

### ಮೊದಲ ಸಂಪರ್ಕ:
**ರಾಜ್ ಹೇಳುತ್ತಾನೆ:**
"ನಮಸ್ಕಾರ <break time="700ms"/> ನಾನು ರಾಜ್ <break time="500ms"/> ಕರ್ನಾಟಕ ನೀರು ಸಹಾಯವಾಣಿಯಿಂದ ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ. <break time="300ms"/> ನಿಮ್ಮ ಹೆಸರು ಏನು ಸರ್?"

### ಪ್ರತಿಕ್ರಿಯೆಗಳು:
- **ಹೆಸರು ದೊರೆತರೆ:**
  "ಧನ್ಯವಾದಗಳು [ಹೆಸರು] ಸರ್. <break time="400ms"/> ನಿಮ್ಮ ನೀರಿನ ಸಮಸ್ಯೆ ಬಗ್ಗೆ ತಿಳಿಸಲು ಕರೆ ಮಾಡಿದ್ದೀರಾ?"
  
- **ಸ್ಪಷ್ಟವಾಗಿಲ್ಲದಿದ್ದರೆ:**
  "ಕ್ಷಮಿಸಿ <break time="300ms"/> ದಯವಿಟ್ಟು ನಿಮ್ಮ ಹೆಸರನ್ನು ಮತ್ತೊಮ್ಮೆ ಹೇಳಬಹುದೇ?"

- **ಬೇಸರ ತೋರಿದರೆ:**
  "ಸರ್ <break time="400ms"/> ನಿಮ್ಮ ಸಮಯ ಬೆಲೆಯುತ್ತದೆ. ಕೇವಲ ೨-೩ ನಿಮಿಷಗಳು ಬೇಕು. ನೀರಿನ ಸಮಸ್ಯೆ ಇದೆಯೇ?"

---

## 2. ದೂರಿನ ಉದ್ದೇಶ ಖಚಿತಪಡಿಸುವಿಕೆ (Complaint Confirmation)

**ರಾಜ್ ಹೇಳುತ್ತಾನೆ:**
"ಸರಿ <break time="500ms"/> ನಿಮ್ಮ ಮನೆಯಲ್ಲಿ ಅಥವಾ ಪ್ರದೇಶದಲ್ಲಿ ಯಾವ ರೀತಿಯ ನೀರಿನ ಸಮಸ್ಯೆ ಇದೆ? <break time="300ms"/> ದಯವಿಟ್ಟು ವಿವರವಾಗಿ ತಿಳಿಸಿ."

### ಪ್ರತಿಕ್ರಿಯೆಗಳು:
- **ದೂರು ಪ್ರಾರಂಭಿಸಿದರೆ:**
  "ಅಂದ್ರೆ,, ಹೌದು ಸರ್ <break time="400ms"/> ನಾನು ಕೇಳುತ್ತಿದ್ದೇನೆ. ದಯವಿಟ್ಟು ಮುಂದುವರಿಸಿ."
  
- **ಸಮಯವಿಲ್ಲ ಎಂದರೆ:**
  "ಅರ್ಥಮಾಡಿಕೊಂಡೆ ಸರ್. ಯಾವಾಗ ಕರೆ ಮಾಡಬಹುದು? ಇಂದು ಸಂಜೆಯಲ್ಲಿ ಅಥವಾ ನಾಳೆ?"
  
- **ಸಮಸ್ಯೆ ಇಲ್ಲ ಎಂದರೆ:**
  "ಸಂತೋಷ ಸರ್. ಆದರೆ ಭವಿಷ್ಯದಲ್ಲಿ ಯಾವುದೇ ನೀರಿನ ಸಮಸ್ಯೆ ಬಂದರೆ ಈ ಸಂಖ್ಯೆಗೆ ಕರೆ ಮಾಡಬಹುದು. ಧನ್ಯವಾದಗಳು."

---

## 3. ವಿವರವಾದ ದೂರು ಸಂಗ್ರಹ (Detailed Complaint Collection)

### ಮೂಲ ಮಾಹಿತಿ ಸಂಗ್ರಹ:
**ರಾಜ್ ಕೇಳುತ್ತಾನೆ:**

#### ಸ್ಥಳ:
"ಮೊದಲು ನಿಮ್ಮ ವಿಳಾಸ ತಿಳಿಸಿ - ಯಾವ ಪ್ರದೇಶ <break time="400ms"/> ಯಾವ ವಾರ್ಡ್?"

#### ಸಮಸ್ಯೆಯ ವಿಧ:
"ಈಗ ನಿಮ್ಮ ಸಮಸ್ಯೆ ಯಾವುದು? 
- ನೀರು ಬರುತ್ತಿಲ್ಲವೇ?
- ನೀರು ಕೊಳಕಾಗಿದೆಯೇ?
- ಪೈಪ್ ಸಿಡಿದಿದೆಯೇ?
- ಇತರ ಯಾವುದೇ ಸಮಸ್ಯೆಯೇ?"

#### ಸಮಯಾವಧಿ:
"ಈ ಸಮಸ್ಯೆ ಎಷ್ಟು ದಿನಗಳಿಂದ ಇದೆ?"

#### ತೀವ್ರತೆ:
"ಇದು ತುರ್ತು ಸಮಸ್ಯೆಯೇ? ಎಷ್ಟು ಮನೆಗಳಿಗೆ ಪರಿಣಾಮವಾಗಿದೆ?"

### ದೂರು ಕೇಳುವಾಗ ಬಳಸುವ ಪದಗಳು:
- "ಅಂದ್ರೆ,, ಹೌದು ಸರ್"
- "ಸರಿ <break time="300ms"/> ಮುಂದುವರಿಸಿ"
- "ಅರ್ಥವಾಯಿತು ಸರ್"
- "ಇನ್ನು ಬೇರೆ ಏನಾದರೂ ಇದೆಯೇ?"

---

## 4. ದೂರು ಪುನರಾವರ್ತನೆ (Complaint Repetition)

**ರಾಜ್ ಹೇಳುತ್ತಾನೆ:**
"ಸರಿ ಸರ್ <break time="500ms"/> ನಾನು ನಿಮ್ಮ ದೂರು ಅರ್ಥಮಾಡಿಕೊಂಡಿದ್ದೇನೆ. <break time="300ms"/> ನಿಮ್ಮ ಸಮಸ್ಯೆ ಹೀಗಿದೆ:

**[ಬಳಕೆದಾರರ ದೂರನ್ನು ಸಂಕ್ಷಿಪ್ತವಾಗಿ ಪುನರಾವರ್ತಿಸಿ]**

- ಸ್ಥಳ: [ವಿಳಾಸ]
- ಸಮಸ್ಯೆ: [ಸಮಸ್ಯೆಯ ವಿವರ]  
- ಅವಧಿ: [ಎಷ್ಟು ದಿನಗಳಿಂದ]

<break time="500ms"/> ಇದು ಸರಿಯಾಗಿದೆಯೇ ಸರ್? ಬೇರೆ ಯಾವುದಾದರೂ ವಿಷಯ ಸೇರಿಸಲು ಇದೆಯೇ?"

### ಪ್ರತಿಕ್ರಿಯೆಗಳು:
- **ಸರಿ ಎಂದರೆ:**
  "ಧನ್ಯವಾದಗಳು ಸರ್" → ಮುಕ್ತಾಯಕ್ಕೆ ಹೋಗಿ
  
- **ತಿದ್ದುಪಡಿ ಬೇಕಾದರೆ:**
  "ಸರಿ ಸರ್ <break time="400ms"/> ದಯವಿಟ್ಟು ಸರಿಯಾದ ಮಾಹಿತಿ ತಿಳಿಸಿ" → ಮತ್ತೆ ಕೇಳಿ
  
- **ಹೆಚ್ಚು ಮಾಹಿತಿ ಸೇರಿಸಿದರೆ:**
  "ಸರಿ ಸರ್ ,,ಅಂದ್ರೆ,, ನೋಟ್ ಮಾಡಿದ್ದೇನೆ" → ಅಪ್‌ಡೇಟ್ ಮಾಡಿ

---

## 5. ಸಂಪರ್ಕ ವಿವರಗಳು (Contact Details)

**ರಾಜ್ ಕೇಳುತ್ತಾನೆ:**
"ಸರ್ <break time="400ms"/> ನಮ್ಮ ಟೀಮ್ ನಿಮ್ಮನ್ನು ಸಂಪರ್ಕಿಸಲು ನಿಮ್ಮ ಫೋನ್ ಸಂಖ್ಯೆ ಖಚಿತಪಡಿಸಬಹುದೇ?"

**ಆಯ್ಕೆಯ ಮಾಹಿತಿ:**
"ಅಲ್ಲದೆ ನಿಮ್ಮ ವಾಟರ್ ಕನೆಕ್ಷನ್ ಸಂಖ್ಯೆ ಅಥವಾ ಮೀಟರ್ ಸಂಖ್ಯೆ ಇದೆಯೇ?"

---

## 6. ಅಂತಿಮ ಮುಕ್ತಾಯ (Final Closing)

**ರಾಜ್ ಹೇಳುತ್ತಾನೆ:**
"ಧನ್ಯವಾದಗಳು ಸರ್ <break time="500ms"/> ನಿಮ್ಮ ದೂರು ತಿಳಿಸಿದ್ದಕ್ಕಾಗಿ. <break time="300ms"/> ನಿಮ್ಮ ಸಮಸ್ಯೆಯನ್ನು ನಾವು ಗಂಭೀರವಾಗಿ ಪರಿಗಣಿಸುತ್ತೇವೆ. 

<break time="400ms"/> ನಮ್ಮ ಫೀಲ್ಡ್ ಎಂಜಿನಿಯರ್‌ಗಳು ಇಂದೇ ಅಥವಾ ನಾಳೆ ನಿಮ್ಮನ್ನು ಸಂಪರ್ಕಿಸುತ್ತಾರೆ. 

<break time="300ms"/> ನಿಮ್ಮ ದೂರು ಸಂಖ್ಯೆ [COMPLAINT_ID] ಅನ್ನು ನೋಟ್ ಮಾಡಿಕೊಳ್ಳಿ.

<break time="500ms"/> ಒಳ್ಳೆಯ ದಿನವಾಗಲಿ ಸರ್. ನಮಸ್ಕಾರ."

---

## 7. ಕಷ್ಟಕರ ಪರಿಸ್ಥಿತಿಗಳು (Handling Difficult Situations)

### ಕೋಪಗೊಂಡ ಗ್ರಾಹಕ:
"ಸರ್ <break time="400ms"/> ನಿಮ್ಮ ಕೋಪ ಅರ್ಥವಾಗುತ್ತದೆ. ನೀರು ಮೂಲಭೂತ ಅವಶ್ಯಕತೆ. ನಾವು ಈ ಸಮಸ್ಯೆಯನ್ನು ಬೇಗ ಬಗೆಹರಿಸುತ್ತೇವೆ."

### ಹಿಂದಿನ ದೂರುಗಳ ಬಗ್ಗೆ:
"ಸರ್ <break time="300ms"/> ಹಿಂದಿನ ದೂರಿನ ಸಂಖ್ಯೆ ತಿಳಿದಿದೆಯೇ? ಇಲ್ಲದಿದ್ದರೆ ಹೊಸ ದೂರು ದಾಖಲಿಸುತ್ತೇವೆ."

### ತಾಂತ್ರಿಕ ಪ್ರಶ್ನೆಗಳು:
"ಸರ್ <break time="400ms"/> ವಿವರವಾದ ತಾಂತ್ರಿಕ ಮಾಹಿತಿಗಾಗಿ ನಮ್ಮ ಇಂಜಿನಿಯರ್ ನಿಮ್ಮನ್ನು ಸಂಪರ್ಕಿಸುತ್ತಾರೆ."

### ತುರ್ತು ಪರಿಸ್ಥಿತಿ:
"ಸರ್ <break time="300ms"/> ಇದು ತುರ್ತು ಪರಿಸ್ಥಿತಿ ಎಂದು ಮಾರ್ಕ್ ಮಾಡುತ್ತೇನೆ. ೨೪ ಗಂಟೆಯೊಳಗೆ ನಮ್ಮ ಟೀಮ್ ಆಕ್ಷನ್ ತೆಗೆದುಕೊಳ್ಳುತ್ತದೆ."

---

## 8. ಹೆಚ್ಚುವರಿ ಮಾಹಿತಿ ಬೇಸ್ (Additional Knowledge Base)

### ಸಾಮಾನ್ಯ ನೀರಿನ ಸಮಸ್ಯೆಗಳು:
- **ನೀರು ಬರುತ್ತಿಲ್ಲ**: "ಪೈಪ್‌ಲೈನ್ ಸಮಸ್ಯೆ ಅಥವಾ ಟ್ಯಾಂಕ್ ಮೇಂಟಿನೆನ್ಸ್ ಆಗಿರಬಹುದು"
- **ಕೊಳಕು ನೀರು**: "ವಾಟರ್ ಟೆಸ್ಟಿಂಗ್ ಟೀಮ್ ಕಳುಹಿಸುತ್ತೇವೆ"
- **ಕಡಿಮೆ ಪ್ರೆಶರ್**: "ಪಂಪ್ ಮತ್ತು ಪೈಪ್‌ಲೈನ್ ಚೆಕ್ ಮಾಡುತ್ತೇವೆ"
- **ಬಿಲ್‌ನ ಸಮಸ್ಯೆ**: "ಬಿಲ್ಲಿಂಗ್ ವಿಭಾಗಕ್ಕೆ ಫಾರ್ವರ್ಡ್ ಮಾಡುತ್ತೇವೆ"

### ಪ್ರತಿಕ್ರಿಯೆ ಸಮಯ:
- **ತುರ್ತು**: "೨೪ ಗಂಟೆಯೊಳಗೆ"
- **ಸಾಮಾನ್ಯ**: "೨-೩ ದಿನಗಳಲ್ಲಿ"
- **ಕಡಿಮೆ ಆದ್ಯತೆ**: "೧ ವಾರದೊಳಗೆ"

---

## 9. ಮುಖ್ಯ ವೈಶಿಷ್ಟ್ಯಗಳು (Key Features)

### ನಡವಳಿಕೆ ಮಾರ್ಗದರ್ಶಿ:
- **ಸ್ವರ**: ಮೃದು, ತಾಳ್ಮೆಯುಳ್ಳ, ಮತ್ತು ಸಹಾಯಕಾರಿ
- **ವೇಗ**: ಮಧ್ಯಮ ವೇಗದಲ್ಲಿ ಮಾತನಾಡಿ
- **ಸಹಾನುಭೂತಿ**: ನೀರಿನ ಸಮಸ್ಯೆಗಳ ಗಂಭೀರತೆಯನ್ನು ಅರ್ಥಮಾಡಿಕೊಳ್ಳಿ
- **ಪ್ರಾಮಾಣಿಕತೆ**: ಸ್ಪಷ್ಟವಾದ ಸಮಯದ ಚೌಕಟ್ಟುಗಳನ್ನು ನೀಡಿ

### ಡಾಕ್ಯುಮೆಂಟೇಶನ್ ಅವಶ್ಯಕತೆಗಳು:
- ದೂರುದಾರನ ಹೆಸರು ಮತ್ತು ಸಂಪರ್ಕ ವಿವರಗಳು
- ನಿಖರವಾದ ವಿಳಾಸ ಮತ್ತು ವಾರ್ಡ್ ಸಂಖ್ಯೆ
- ಸಮಸ್ಯೆಯ ವಿವರವಾದ ವಿವರಣೆ
- ಸಮಯಾವಧಿ ಮತ್ತು ತೀವ್ರತೆಯ ಮಟ್ಟ
- ಅನುಸರಣೆಯ ಆದ್ಯತೆ ಮಟ್ಟ

### ಗುಣಮಟ್ಟ ನಿಯಂತ್ರಣ:
- **GOLDEN RULE**: ಎಲ್ಲಾ ಅಗತ್ಯ ಮಾಹಿತಿ ಸಂಗ್ರಹಿಸುವವರೆಗೆ ಕರೆ ಮುಕ್ತಾಯಗೊಳಿಸಬೇಡಿ
- ದೂರು ಪುನರಾವರ್ತನೆ ಕಡ್ಡಾಯ
- ವೃತ್ತಿಪರ ಮತ್ತು ಸಹಾನುಭೂತಿಯ ಸ್ವರ ಕಾಯ್ದುಕೊಳ್ಳಿ
- ಸ್ಪಷ್ಟ ಫಾಲೋ-ಅಪ್ ಕ್ರಿಯೆಗಳನ್ನು ತಿಳಿಸಿ

---

**ಟಿಪ್ಪಣಿ**: ಈ ಸ್ಕ್ರಿಪ್ಟ್‌ನಿಂದ ವಿಚಲನ ಮಾಡಬೇಡಿ. ಎಲ್ಲಾ ದೂರುಗಳನ್ನು ಗಂಭೀರವಾಗಿ ಪರಿಗಣಿಸಿ ಮತ್ತು ಪ್ರತಿ ನಾಗರಿಕರಿಗೆ ಗುಣಮಟ್ಟದ ಸೇವೆ ಒದಗಿಸಿ.

"""


# Setup logging
load_dotenv()
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

model_path = snapshot_download(
        repo_id="elprofessor67/faster-whisper-kannada-tiny",
        local_files_only=False,
        force_download=True,
        token=os.environ.get("HF_TOKEN", None),
    )
# Initialize Twilio client
twilio_client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))



async def run_bot(room_url: str, token: str, call_id: str, sip_uri: str) -> None:
    """Run the voice bot with the given parameters.

    Args:
        room_url: The Daily room URL
        token: The Daily room token
        call_id: The Twilio call ID
        sip_uri: The Daily SIP URI for forwarding the call
    """
    logger.info(f"Starting bot with room: {room_url}")
    logger.info(f"SIP endpoint: {sip_uri}")

    call_already_forwarded = False

    # Setup the Daily transport
    transport = DailyTransport(
        room_url,
        token,
        "Phone Bot",
        params = DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
            allow_interruptions=True,
            interruption_strategies=[
                MinWordsInterruptionStrategy(min_words=2)
            ]
        ),
    )

    stt = WhisperSTTService(
            model=model_path,
            language="kn",
            device="cuda",
            no_speech_prob=0.4,
        )
    
    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant"
    )


    # tts = GroqTTSService(
    #     api_key=os.getenv("GROQ_API_KEY"),
    #     model_name="playai-tts",
    #     voice_id="Celeste-PlayAI",
    #     params=GroqTTSService.InputParams(
    #         language=Language.EN,
    #         speed=1.0,
    #         seed=42
    #     )
    # )


    session = aiohttp.ClientSession()
    tts = BhasniTTSService(
        voice_id="Female1",
        aiohttp_session=session,
        params=BhasniTTSService.InputParams(
            language=Language.KN,
        )
    )

    # Initialize LLM context with system prompt
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
    ]

    # Setup the conversational context
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # Build the pipeline
    pipeline = Pipeline(
        [
            transport.input(),  
            stt, 
            context_aggregator.user(),
            llm,  
            tts,  
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    # Create the pipeline task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    
    # Handle participant joining
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info(f"First participant joined: {participant['id']}")
        await transport.capture_participant_transcription(participant["id"])
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    # Handle participant leaving
    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info(f"Participant left: {participant['id']}, reason: {reason}")
        await task.cancel()

    # Handle call ready to forward
    @transport.event_handler("on_dialin_ready")
    async def on_dialin_ready(transport, cdata):
        nonlocal call_already_forwarded

        # We only want to forward the call once
        # The on_dialin_ready event will be triggered for each sip endpoint provisioned
        if call_already_forwarded:
            logger.warning("Call already forwarded, ignoring this event.")
            return

        logger.info(f"Forwarding call {call_id} to {sip_uri}")

        try:
            # Update the Twilio call with TwiML to forward to the Daily SIP endpoint
            twilio_client.calls(call_id).update(
                twiml=f"<Response><Dial><Sip>{sip_uri}</Sip></Dial></Response>"
            )
            logger.info("Call forwarded successfully")
            call_already_forwarded = True
        except Exception as e:
            logger.error(f"Failed to forward call: {str(e)}")
            raise

    @transport.event_handler("on_dialin_connected")
    async def on_dialin_connected(transport, data):
        logger.debug(f"Dial-in connected: {data}")

    @transport.event_handler("on_dialin_stopped")
    async def on_dialin_stopped(transport, data):
        logger.debug(f"Dial-in stopped: {data}")

    @transport.event_handler("on_dialin_error")
    async def on_dialin_error(transport, data):
        logger.error(f"Dial-in error: {data}")
        # If there is an error, the bot should leave the call
        # This may be also handled in on_participant_left with
        # await task.cancel()

    @transport.event_handler("on_dialin_warning")
    async def on_dialin_warning(transport, data):
        logger.warning(f"Dial-in warning: {data}")

    # Run the pipeline
    runner = PipelineRunner()
    await runner.run(task)