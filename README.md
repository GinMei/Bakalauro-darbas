# Bakalauro-darbas
3D ŽAIDIMŲ SCENŲ GENERAVIMAS TAIKANT DIRBTINIO INTELEKTO AGENTUS

Žaidimų variklio projektų konverteris
Įrankis, automatiškai konvertuojantis žaidimų projektus tarp Godot 4.5 ir Unity 6000.3.9f1 variklinų. Konvertavimo metu perkeliamos scenos, objektų hierarchijos, komponentai, resursai ir programinis kodas. Scenos klasifikavimui naudojamas vietinis DI modelis (Ollama / qwen3), o programinio kodo konvertavimui – Gemini API.

# Reikalavimai
Python 3.10+
Ollama su qwen3 modeliu (scenų klasifikavimui)
Gemini API raktas (programinio kodo konvertavimui)
Instaliavimas

# 1. Klonuoti repozitoriją
git clone <repo-url>
cd ConverterRepository/Converter_project_files

# 2. Sukurti virtualią aplinką ir įdiegti priklausomybes
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 3. Nustatyti aplinkos kintamuosius
set GEMINI_API_KEY=<jūsų_raktas>      # Windows
# arba
export GEMINI_API_KEY=<jūsų_raktas>   # Linux/macOS

# 4. Paleisti Ollama su qwen3 modeliu
ollama run qwen3
Paleidimas

uvicorn api:app --reload --port 8000
Naršyklėje atidaryti: http://127.0.0.1:8000