class SpeechEngine {
    constructor() {
        this.synth = window.speechSynthesis;
        this.rate = 1.0;
        this.volume = 1.0;
        this.pitch = 1.0;
        this.audioCtx = null;
    }

    initAudio() {
        try {
            if (!this.audioCtx) {
                const AudioContext = window.AudioContext || window.webkitAudioContext;
                if (AudioContext) {
                    this.audioCtx = new AudioContext();
                }
            }
            if (this.audioCtx && this.audioCtx.state === "suspended") {
                this.audioCtx.resume();
            }
        } catch (e) {}
    }

    playTone(frequency = 440, durationSec = 0.15, type = "sine") {
        try {
            this.initAudio();
            if (!this.audioCtx) return;
            const osc = this.audioCtx.createOscillator();
            const gain = this.audioCtx.createGain();
            osc.type = type;
            osc.frequency.value = frequency;
            gain.gain.setValueAtTime(0.25, this.audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, this.audioCtx.currentTime + durationSec);
            osc.connect(gain);
            gain.connect(this.audioCtx.destination);
            osc.start();
            osc.stop(this.audioCtx.currentTime + durationSec);
        } catch (e) {
            // Audio context not allowed until user interaction
        }
    }

    playStartCue() {
        this.playTone(523.25, 0.1); // C5
        setTimeout(() => this.playTone(659.25, 0.15), 120); // E5
    }

    playStopCue() {
        this.playTone(659.25, 0.1); // E5
        setTimeout(() => this.playTone(440.0, 0.18), 120); // A4
    }

    playAlertCue() {
        this.playTone(880.0, 0.2, "sawtooth"); // Urgent high alert
    }

    speak(text, priority = "MEDIUM") {
        if (!text || !this.synth) return;

        try {
            // Critical alerts cancel lower-priority speech immediately
            if (priority === "CRITICAL") {
                this.playAlertCue();
                this.synth.cancel();
            } else if (this.synth.speaking) {
                // Do not backlog lower-priority speech while voice is busy
                return;
            }

            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = this.rate;
            utterance.volume = this.volume;
            utterance.pitch = this.pitch;

            this.synth.speak(utterance);
        } catch (e) {
            console.warn("Speech synthesis error:", e);
        }
    }

    stop() {
        if (this.synth) {
            this.synth.cancel();
        }
    }
}

window.speechEngine = new SpeechEngine();
