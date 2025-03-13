#!/usr/bin/env python
"""
Script to generate a test audio file simulating an advisory meeting in Swedish that can be
processed by the Speech2StructuredDoc application according to the Pydantic model.
"""

import os
import sys
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import datetime

# Add the src directory to the path so we can import modules from there
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load environment variables
load_dotenv()

# Configuration
SPEECH_REGION = os.getenv("SPEECH_REGION", "swedencentral")
SPEECH_API_KEY = os.getenv("SPEECH_API_KEY")

def generate_test_audio(output_filename="test_advisory_meeting.wav"):
    """
    Generate a test audio file simulating an advisory meeting in Swedish with all fields
    required by the Pydantic model.
    """
    # Create the advisory meeting script with all required fields in Swedish
    today = datetime.date.today().strftime("%d %B, %Y").replace("March", "mars")
    
    meeting_script = f"""
    Finansiellt rådgivningsmöte med Söderberg & Partners AB den {today}.
    
    Deltagare i dagens möte är Maria Johansson från Söderberg & Partners, 
    och kunderna Erik Andersson och Lena Karlsson från Volvo Group.
    
    Maria: God morgon allihopa. Tack för att ni deltar i vårt kvartalsvisa finansiella rådgivningsmöte.
    Idag ska vi gå igenom er nuvarande investeringsportfölj, diskutera marknadstrender och planera för 
    det kommande räkenskapsåret.
    
    Erik: Tack Maria. Vi är särskilt intresserade av att förstå hur den senaste marknadsvolatiliteten 
    kan påverka vår pensionsplanering.
    
    Maria: Det är en bra poäng, Erik. Låt oss börja med en översikt av er nuvarande portföljallokering. 
    För närvarande har ni 60% i aktiefonder, 30% i räntebärande värdepapper och 10% i alternativa 
    investeringar. Det totala portföljvärdet är 2,5 miljoner kronor, vilket representerar en ökning med 
    4,2% sedan vårt senaste möte.
    
    Lena: Det låter positivt. Hur ser det ut med våra ESG-investeringar? Vi var intresserade av att öka 
    vår allokering till hållbara fonder.
    
    Maria: Ja, Lena. För närvarande är 25% av er aktieallokering i ESG-fokuserade fonder. Baserat på era 
    preferenser rekommenderar jag att öka detta till 40%, med särskilt fokus på Nordic Sustainable 
    Equity Fund som har visat stark prestation samtidigt som den upprätthåller strikta hållbarhetskriterier.
    
    Erik: Det låter vettigt. Hur är det med den räntebärande delen med tanke på den nuvarande räntemiljön?
    
    Maria: Bra fråga. Med de nuvarande räntetrenderna föreslår jag att vi justerar er räntestrategi för att 
    inkludera fler kort- till medellånga obligationer för att minska ränterisken. Detta skulle innebära 
    en omallokering av cirka 10% av era räntebärande innehav från långsiktiga till kortare instrument.
    
    Lena: Och hur ser det ut med vår pensionsplaneringstidslinje? Vi planerar fortfarande att Erik går i 
    pension om fem år och jag om sju år.
    
    Maria: Baserat på er nuvarande sparkvot och portföljutveckling är ni på rätt spår för att nå era 
    pensionsmål. Jag rekommenderar dock att öka era månatliga bidrag med 1000 kronor för att ge en 
    ytterligare buffert, särskilt med tanke på marknadsosäkerheten som du nämnde, Erik.
    
    Erik: Det låter rimligt. Vi kan hantera den ökningen.
    
    Maria: Utmärkt. Låt mig sammanfatta huvudpunkterna från dagens diskussion:
    1. Öka ESG-allokeringen från 25% till 40% av aktieinnehaven
    2. Flytta 10% av räntebärande värdepapper från långsiktiga till kortare instrument
    3. Öka månatliga bidrag med 1000 kronor
    4. Behåll nuvarande pensionsplan med Erik som går i pension om 5 år och Lena om 7 år
    
    För våra nästa steg:
    1. Jag kommer att förbereda omallokeringsdokumenten för ert godkännande senast nästa fredag
    2. Vi bör schemalägga ett uppföljningsmöte i början av april för att granska Q1-resultatet
    3. Jag kommer att skicka information om skattekonsekvenserna av de föreslagna ändringarna
    4. Låt oss sätta upp ett möte med vår pensionsspecialist för att granska era pensionsplaner i detalj
    
    Täcker denna sammanfattning allt vi diskuterade idag?
    
    Lena: Ja, det täcker allt från mitt perspektiv.
    
    Erik: Jag håller med. Detta ger oss en tydlig väg framåt.
    
    Maria: Underbart. Tack för er tid idag. Jag kommer att höra av mig med dokumenten och för att 
    schemalägga våra uppföljningsmöten.
    
    Mötet avslutades klockan 11:45.
    """
    
    try:
        print(f"Genererar testljudfil: {output_filename}")
        
        # Create speech configuration with API key
        speech_config = speechsdk.SpeechConfig(subscription=SPEECH_API_KEY, region=SPEECH_REGION)
        
        # Use a Swedish voice
        speech_config.speech_synthesis_voice_name = "sv-SE-SofieNeural"
        
        # Configure audio output
        file_config = speechsdk.audio.AudioOutputConfig(filename=output_filename)
        
        # Create speech synthesizer
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=file_config)
        
        print("Syntetiserar tal...")
        result = synthesizer.speak_text_async(meeting_script).get()
        
        # Check result
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"Testljudfilen har skapats framgångsrikt: {output_filename}")
            print(f"Denna ljudfil innehåller ett rådgivningsmöte med alla fält som krävs av Pydantic-modellen:")
            print(f"- Kundnamn: Volvo Group")
            print(f"- Mötesdatum: {today}")
            print(f"- Deltagare: Maria Johansson, Erik Andersson, Lena Karlsson")
            print(f"- Huvudpunkter: Portföljöversyn, ökning av ESG-investeringar, justering av räntebärande värdepapper, pensionsplanering")
            print(f"- Åtgärdspunkter: Förbereda omallokeringsdokument, schemalägga uppföljningsmöte, skicka skatteinformation, ordna möte med pensionsspecialist")
            return True
        else:
            print(f"Talsyntes misslyckades: {result.reason}")
            return False
            
    except Exception as e:
        print(f"Fel vid generering av testljud: {str(e)}")
        return False

if __name__ == "__main__":
    print("Genererar testljudfil för rådgivningsmöte...")
    success = generate_test_audio()
    if success:
        print("Generering av testljud slutfördes framgångsrikt!")
    else:
        print("Generering av testljud misslyckades.")
        sys.exit(1)
