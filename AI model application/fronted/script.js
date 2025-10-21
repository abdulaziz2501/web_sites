/**
 * YOLOv8 Real-Time Object Detection
 * Telefon kamerasi bilan ishlash va detection natijalarini ko'rsatish
 */

// Global o'zgaruvchilar
let video;
let canvas;
let ctx;
let stream;
let detectionInterval;
let currentFacingMode = 'environment'; // 'user' yoki 'environment'
let isDetecting = false;

// API konfiguratsiyasi
const API_URL = window.location.origin; // Hozirgi host (server IP)
let detectionFPS = 10; // sekundiga nechta detection
let confidenceThreshold = 0.25;

// Statistika
let frameCount = 0;
let lastFrameTime = Date.now();
let detectionLatency = 0;

/**
 * Sahifa yuklanganida ishga tushadi
 */
document.addEventListener('DOMContentLoaded', () => {
    initializeElements();
    setupEventListeners();
    checkServerHealth();
});

/**
 * HTML elementlarini olish
 */
function initializeElements() {
    video = document.getElementById('video');
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
}

/**
 * Event listenerlarni o'rnatish
 */
function setupEventListeners() {
    document.getElementById('startBtn').addEventListener('click', startCamera);
    document.getElementById('stopBtn').addEventListener('click', stopCamera);
    document.getElementById('switchCamera').addEventListener('click', switchCamera);
    
    // FPS slider
    document.getElementById('fpsSlider').addEventListener('input', (e) => {
        detectionFPS = parseInt(e.target.value);
        document.getElementById('fpsValue').textContent = detectionFPS;
        
        // Agar detection boshlangan bo'lsa, restart qilish
        if (isDetecting) {
            stopDetection();
            startDetection();
        }
    });
    
    // Confidence slider
    document.getElementById('confidenceSlider').addEventListener('input', (e) => {
        confidenceThreshold = parseInt(e.target.value) / 100;
        document.getElementById('confidenceValue').textContent = confidenceThreshold.toFixed(2);
    });
}

/**
 * Server holatini tekshirish
 */
async function checkServerHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            updateStatus('online', 'Server tayyor ✅');
            document.getElementById('modelName').textContent = data.model;
        }
    } catch (error) {
        updateStatus('offline', 'Serverga ulanib bo\'lmadi ❌');
        console.error('Server health check xatosi:', error);
    }
}

/**
 * Status yangilash
 */
function updateStatus(status, text) {
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.getElementById('statusText');
    
    statusDot.className = `status-dot ${status}`;
    statusText.textContent = text;
}

/**
 * Kamerani yoqish
 */
async function startCamera() {
    try {
        updateStatus('connecting', 'Kamera yoqilmoqda...');
        
        // Kamera ruxsatini so'rash
        const constraints = {
            video: {
                facingMode: currentFacingMode,
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        };
        
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        
        // Video metadata yuklanganida
        video.onloadedmetadata = () => {
            video.play();
            
            // Canvas o'lchamini sozlash
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            // Buttonlarni faollashtirish
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('switchCamera').disabled = false;
            
            updateStatus('online', 'Kamera faol 🎥');
            
            // Detection boshlash
            startDetection();
        };
        
    } catch (error) {
        console.error('Kamera xatosi:', error);
        updateStatus('offline', 'Kamera ochilmadi ❌');
        alert('Kamera ruxsati berilmagan yoki kamera topilmadi!');
    }
}

/**
 * Kamerani to'xtatish
 */
function stopCamera() {
    stopDetection();
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }
    
    // Canvas tozalash
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Buttonlarni qayta sozlash
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    document.getElementById('switchCamera').disabled = true;
    
    // Overlay tozalash
    document.getElementById('overlay').innerHTML = '';
    document.getElementById('detectionsList').innerHTML = '';
    
    updateStatus('offline', 'Kamera o\'chirildi');
}

/**
 * Kamerani almashtirish (old/orqa)
 */
async function switchCamera() {
    currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
    stopCamera();
    await startCamera();
}

/**
 * Object detection boshlash
 */
function startDetection() {
    isDetecting = true;
    const intervalTime = 1000 / detectionFPS; // ms
    
    detectionInterval = setInterval(() => {
        captureAndDetect();
    }, intervalTime);
}

/**
 * Detection to'xtatish
 */
function stopDetection() {
    isDetecting = false;
    if (detectionInterval) {
        clearInterval(detectionInterval);
    }
}

/**
 * Frame capture va detection yuborish
 */
async function captureAndDetect() {
    if (!video.videoWidth || !video.videoHeight) return;
    
    // Video frameni canvasga chizish
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Canvas'ni base64 ga aylantirish
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    
    // Detectionni yuborish
    const startTime = Date.now();
    
    try {
        const response = await fetch(`${API_URL}/detect`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageData })
        });
        
        const data = await response.json();
        
        // Latency hisoblash
        detectionLatency = Date.now() - startTime;
        document.getElementById('latency').textContent = `${detectionLatency}ms`;
        
        if (data.success) {
            // Natijalarni ko'rsatish
            displayDetections(data.detections);
            
            // Statistika yangilash
            updateStatistics(data.count);
        }
        
    } catch (error) {
        console.error('Detection xatosi:', error);
        updateStatus('offline', 'Serverga ulanishda xato ❌');
    }
}

/**
 * Detection natijalarini ko'rsatish
 */
function displayDetections(detections) {
    const overlay = document.getElementById('overlay');
    const detectionsList = document.getElementById('detectionsList');
    
    // Overlay tozalash
    overlay.innerHTML = '';
    
    // Detections listini tozalash
    detectionsList.innerHTML = '';
    
    // Filtr: confidence threshold
    const filtered = detections.filter(det => det.confidence >= confidenceThreshold);
    
    // Har bir detection uchun
    filtered.forEach((det, index) => {
        // Bounding box chizish
        const box = det.bbox;
        const boxDiv = document.createElement('div');
        boxDiv.className = 'detection-box';
        boxDiv.style.left = `${box.x1}px`;
        boxDiv.style.top = `${box.y1}px`;
        boxDiv.style.width = `${box.x2 - box.x1}px`;
        boxDiv.style.height = `${box.y2 - box.y1}px`;
        
        // Label qo'shish
        const label = document.createElement('div');
        label.className = 'detection-label';
        label.textContent = `${det.class} ${(det.confidence * 100).toFixed(0)}%`;
        boxDiv.appendChild(label);
        
        overlay.appendChild(boxDiv);
        
        // Ro'yxatga qo'shish
        const listItem = document.createElement('div');
        listItem.className = 'detection-item';
        listItem.innerHTML = `
            <span class="detection-name">${det.class}</span>
            <span class="detection-confidence">${(det.confidence * 100).toFixed(0)}%</span>
        `;
        detectionsList.appendChild(listItem);
    });
}

/**
 * Statistika yangilash
 */
function updateStatistics(objectCount) {
    // Obyektlar soni
    document.getElementById('objectCount').textContent = objectCount;
    
    // FPS hisoblash
    frameCount++;
    const currentTime = Date.now();
    const elapsed = currentTime - lastFrameTime;
    
    if (elapsed >= 1000) { // Har sekundda yangilash
        const actualFps = Math.round(frameCount / (elapsed / 1000));
        document.getElementById('actualFps').textContent = actualFps;
        
        frameCount = 0;
        lastFrameTime = currentTime;
    }
}

/**
 * Server info olish
 */
async function getServerInfo() {
    try {
        const response = await fetch(`${API_URL}/`);
        const data = await response.json();
        document.getElementById('serverIp').textContent = data.server_ip;
    } catch (error) {
        console.error('Server info olishda xato:', error);
    }
}

// Server info yuklanishi
getServerInfo();
