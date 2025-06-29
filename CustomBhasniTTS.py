from typing import AsyncGenerator, Optional
import io
import aiohttp
from loguru import logger
from pydub import AudioSegment
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts


def language_to_bhasni_language(language: Language) -> Optional[str]:
    """Convert Pipecat Language enum to Bhasni language codes."""
    LANGUAGE_MAP = {
        Language.AS: "Assamese",  # Assamese
        Language.BN: "Bengali",  # Bengali
        Language.BO: "Bodo",  # Bodo
        "DOG": "Dogri",  # Dogri
        Language.EN: "English",  # English (India)
        Language.GU: "Gujarati",  # Gujarati
        Language.HI: "Hindi",  # Hindi
        Language.KN: "Kannada",  # Kannada
        "KS": "Kashmiri",  # Kashmiri
        "KK": "Konkani",  # Konkani
        "MAI": "Maithili",  # Maithili
        Language.ML: "Malayalam",  # Malayalam
        "MNI": "Manipuri",  # Manipuri
        Language.MR: "Marathi",  # Marathi
        Language.NE: "Nepali",  # Nepali
        Language.OR: "Odia",  # Odia
        Language.PA: "Punjabi",  # Punjabi
        Language.SA: "Sanskrit",  # Sanskrit
        "SAN": "Santali",  # Santali
        Language.SD: "Sindhi",  # Sindhi
        Language.TA: "Tamil",  # Tamil
        Language.TE: "Telugu",  # Telugu
        Language.UR: "Urdu",  # Urdu
    }

    return LANGUAGE_MAP.get(language)


class BhasniTTSService(TTSService):
    """Text-to-Speech service using Bhasni AI's API.

    Converts text to speech using Bhasni AI's TTS models with support for multiple
    Indian languages. Provides control over voice characteristics like pitch, pace,
    and loudness.

    Args:
        api_key: Bhasni AI API subscription key.
        voice_id: Speaker voice ID (e.g., "Female1", "Male1").
        model: TTS model to use ("bhasni").
        aiohttp_session: Shared aiohttp session for making requests.
        base_url: Bhasni AI API base URL.
        sample_rate: Audio sample rate in Hz (8000, 16000, 22050, 24000).
        params: Additional voice and preprocessing parameters.

    Example:
        ```python
        tts = BhasniTTSService(
            api_key="your-api-key",
            voice_id="Female1",
            model="bhasni",
            aiohttp_session=session,
            params=BhasniTTSService.InputParams(
                language=Language.HI
            )
        )
        ```
    """

    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        voice_id: str = "Female1",
        aiohttp_session: aiohttp.ClientSession,
        base_url: str = "https://tts.bhashini.ai/v1",
        params: Optional[InputParams] = None,
        model: Optional[str] = "bhasni",
        sample_rate: Optional[int] = 16000,  # Default to 16kHz
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or BhasniTTSService.InputParams()

        self._api_key = api_key
        self._base_url = base_url
        self._session = aiohttp_session

        self._settings = {
            "language": self.language_to_service_language(params.language)
            if params.language
            else "English"
        }

        self.set_model_name(model)
        self.set_voice(voice_id)

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        return language_to_bhasni_language(language)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._settings["sample_rate"] = self.sample_rate

    def _convert_mp3_to_pcm(self, mp3_data: bytes) -> bytes:
        """Convert MP3 data to raw PCM data."""
        try:
            # Load MP3 from bytes
            audio_segment = AudioSegment.from_mp3(io.BytesIO(mp3_data))
            
            # Convert to target format
            # Ensure we match the expected sample rate and format
            audio_segment = audio_segment.set_frame_rate(self.sample_rate)
            audio_segment = audio_segment.set_channels(1)  # Mono
            audio_segment = audio_segment.set_sample_width(2)  # 16-bit
            
            # Get raw PCM data
            pcm_data = audio_segment.raw_data
            
            return pcm_data
            
        except Exception as e:
            logger.error(f"Error converting MP3 to PCM: {e}")
            raise

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()

            payload = {
                "text": text,
                "language": self._settings["language"],
                "voiceName": self._voice_id
            }

            headers = {
                "Content-Type": "application/json",
            }

            if self._api_key:
                headers["x-api-key"] = self._api_key

            url = f"{self._base_url}/synthesize"

            yield TTSStartedFrame()

            async with self._session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Bhasni API error: {error_text}")
                    await self.push_error(ErrorFrame(f"Bhasni API error: {error_text}"))
                    return

                # Read the MP3 response
                mp3_data = await response.read()

            await self.start_tts_usage_metrics(text)

            # Convert MP3 to raw PCM data
            pcm_data = self._convert_mp3_to_pcm(mp3_data)
            # Create audio frame with raw PCM data
            frame = TTSAudioRawFrame(
                audio=pcm_data,
                sample_rate=self.sample_rate,
                num_channels=1,
            )

            yield frame

        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error(ErrorFrame(f"Error generating TTS: {e}"))
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()

    # Alternative method if you specifically need chunked output
    @traced_tts
    async def run_tts_chunked(self, text: str, chunk_duration_ms: int = 100) -> AsyncGenerator[Frame, None]:
        """Generate TTS with chunked audio output for streaming."""
        logger.debug(f"{self}: Generating chunked TTS [{text}]")

        try:
            await self.start_ttfb_metrics()

            payload = {
                "text": text,
                "language": self._settings["language"],
                "voiceName": self._voice_id
            }

            headers = {
                "Content-Type": "application/json",
            }

            if self._api_key:
                headers["x-api-key"] = self._api_key

            url = f"{self._base_url}/synthesize"

            yield TTSStartedFrame()

            async with self._session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Bhasni API error: {error_text}")
                    await self.push_error(ErrorFrame(f"Bhasni API error: {error_text}"))
                    return

                mp3_data = await response.read()

            await self.start_tts_usage_metrics(text)

            # Convert MP3 to PCM and chunk it
            try:
                audio_segment = AudioSegment.from_mp3(io.BytesIO(mp3_data))
                audio_segment = audio_segment.set_frame_rate(self.sample_rate)
                audio_segment = audio_segment.set_channels(1)
                audio_segment = audio_segment.set_sample_width(2)

                # Split into chunks
                for i in range(0, len(audio_segment), chunk_duration_ms):
                    chunk = audio_segment[i:i + chunk_duration_ms]
                    chunk_pcm = chunk.raw_data
                    
                    if len(chunk_pcm) > 0:  # Only yield non-empty chunks
                        frame = TTSAudioRawFrame(
                            audio=chunk_pcm,
                            sample_rate=self.sample_rate,
                            num_channels=1,
                        )
                        yield frame

            except Exception as e:
                logger.error(f"Error processing MP3 audio: {e}")
                await self.push_error(ErrorFrame(f"Error processing audio: {e}"))
                return

        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error(ErrorFrame(f"Error generating TTS: {e}"))
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()