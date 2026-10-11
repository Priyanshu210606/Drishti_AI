class WebSocketClient {
    constructor(endpoint = "/ws") {
        this.endpoint = endpoint;
        this.ws = null;
        this.isConnected = false;
        this.onMessageCallback = null;
        this.onStatusChangeCallback = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.shouldReconnect = true;
    }

    connect() {
        if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
            return;
        }

        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const url = `${protocol}//${window.location.host}${this.endpoint}`;

        this._notifyStatus("connecting", "Connecting to server...");

        try {
            this.ws = new WebSocket(url);
            this.ws.binaryType = "blob";

            this.ws.onopen = () => {
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this._notifyStatus("ready", "System Ready");
            };

            this.ws.onmessage = (event) => {
                if (this.onMessageCallback && typeof event.data === "string") {
                    try {
                        const parsed = JSON.parse(event.data);
                        this.onMessageCallback(parsed);
                    } catch (e) {
                        console.error("Malformed WebSocket message:", e);
                    }
                }
            };

            this.ws.onclose = () => {
                this.isConnected = false;
                this._notifyStatus("error", "Disconnected");
                if (this.shouldReconnect) {
                    this._scheduleReconnect();
                }
            };

            this.ws.onerror = (err) => {
                console.error("WebSocket error:", err);
                this.ws.close();
            };
        } catch (e) {
            console.error("Failed creating WebSocket:", e);
            this._scheduleReconnect();
        }
    }

    sendFrame(blob) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(blob);
        }
    }

    sendConfig(configObj) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: "config", ...configObj }));
        }
    }

    _scheduleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.min(5000, 1000 * Math.pow(1.5, this.reconnectAttempts));
            this._notifyStatus("connecting", `Reconnecting (${this.reconnectAttempts})...`);
            setTimeout(() => this.connect(), delay);
        } else {
            this._notifyStatus("error", "Server unavailable");
        }
    }

    _notifyStatus(status, text) {
        if (this.onStatusChangeCallback) {
            this.onStatusChangeCallback(status, text);
        }
    }

    disconnect() {
        this.shouldReconnect = false;
        if (this.ws) {
            this.ws.close();
        }
    }
}

window.WebSocketClient = WebSocketClient;
