class CameraManager {
    constructor(videoElement, overlayCanvas) {
        this.video = videoElement;
        this.canvas = overlayCanvas;
        this.ctx = overlayCanvas.getContext("2d");
        this.stream = null;
        this.currentFacingMode = "environment"; // Default to rear camera for forward navigation
        this.captureCanvas = document.createElement("canvas");
        this.captureCtx = this.captureCanvas.getContext("2d");
        this.targetWidth = 640;
        this.targetHeight = 480;
        this.captureCanvas.width = this.targetWidth;
        this.captureCanvas.height = this.targetHeight;
        this.canvas.width = this.targetWidth;
        this.canvas.height = this.targetHeight;
        this.captureInterval = null;
    }

    _syncDimensions() {
        const w = this.video.videoWidth || this.targetWidth;
        const h = this.video.videoHeight || this.targetHeight;
        this.canvas.width = w;
        this.canvas.height = h;
        this.captureCanvas.width = w;
        this.captureCanvas.height = h;
    }

    async startCamera() {
        if (this.stream) {
            this.stopCamera();
        }

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("Camera API is not supported in this browser context (requires localhost or HTTPS).");
        }

        this.video.muted = true;
        this.video.playsInline = true;

        const constraints = {
            audio: false,
            video: {
                facingMode: { ideal: this.currentFacingMode },
                width: { ideal: this.targetWidth },
                height: { ideal: this.targetHeight }
            }
        };

        try {
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;
            await this.video.play();
            this._syncDimensions();
            this.video.addEventListener("loadedmetadata", () => this._syncDimensions(), { once: true });
            return true;
        } catch (err) {
            console.warn("Preferred camera constraint failed, attempting generic fallback:", err);
            try {
                this.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                this.video.srcObject = this.stream;
                await this.video.play();
                this._syncDimensions();
                this.video.addEventListener("loadedmetadata", () => this._syncDimensions(), { once: true });
                return true;
            } catch (fallbackErr) {
                console.error("Camera fallback failed:", fallbackErr);
                throw fallbackErr;
            }
        }
    }

    async switchCamera() {
        this.currentFacingMode = (this.currentFacingMode === "environment") ? "user" : "environment";
        if (this.stream) {
            await this.startCamera();
        }
        return this.currentFacingMode;
    }

    startFrameCapture(fps, onFrameReady) {
        if (this.captureInterval) {
            clearInterval(this.captureInterval);
        }

        const intervalMs = Math.round(1000 / fps);
        this.captureInterval = setInterval(() => {
            if (!this.stream || this.video.readyState < 2) {
                return;
            }

            this.captureCtx.drawImage(this.video, 0, 0, this.captureCanvas.width, this.captureCanvas.height);
            this.captureCanvas.toBlob((blob) => {
                if (blob && onFrameReady) {
                    onFrameReady(blob);
                }
            }, "image/jpeg", 0.75);

        }, intervalMs);
    }

    stopFrameCapture() {
        if (this.captureInterval) {
            clearInterval(this.captureInterval);
            this.captureInterval = null;
        }
        this.clearOverlay();
    }

    stopCamera() {
        this.stopFrameCapture();
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        if (this.video) {
            this.video.srcObject = null;
        }
    }

    drawDetections(objects) {
        if (!this.ctx) return;
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        const w = this.canvas.width;
        const h = this.canvas.height;

        objects.forEach(obj => {
            const x1 = obj.bbox.x1 * w;
            const y1 = obj.bbox.y1 * h;
            const bw = obj.bbox.width * w;
            const bh = obj.bbox.height * h;

            // Highlight path corridor obstacles in bright orange/red
            const color = obj.is_in_path ? "#ff1744" : "#00e676";
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 3;
            this.ctx.strokeRect(x1, y1, bw, bh);

            // Distance & label tag
            this.ctx.fillStyle = "rgba(0, 0, 0, 0.75)";
            const tagText = `${obj.label} • ${obj.distance_m}m`;
            this.ctx.font = "bold 14px sans-serif";
            const textWidth = this.ctx.measureText(tagText).width;
            this.ctx.fillRect(x1, Math.max(0, y1 - 24), textWidth + 12, 22);

            this.ctx.fillStyle = color;
            this.ctx.fillText(tagText, x1 + 6, Math.max(16, y1 - 8));
        });
    }

    clearOverlay() {
        if (this.ctx) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
    }
}

window.CameraManager = CameraManager;
