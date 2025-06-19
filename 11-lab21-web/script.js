class ComputerVisionDemo {
    constructor() {
        this.models = {};
        this.webcamStream = null;
        this.isDetecting = false;
        this.init();
    }

    async init() {
        await this.loadModels();
        this.setupEventListeners();
        console.log('ok inicio!');
    }

    async loadModels() {
        try {
            console.log('Carregando modelos...');
            this.models.mobilenet = await mobilenet.load();
            console.log('MobileNet carregado');
            this.models.cocoSsd = await cocoSsd.load();
            console.log('COCO-SSD carregado');
        } catch (error) {
            console.error('Erro ao carregar modelos:', error);
            const classificationResults = document.getElementById('classificationResults');
            if (classificationResults) {
                classificationResults.innerHTML = `<p class="error">Erro ao carregar modelos: ${error.message}</p>`;
            }
        }
    }

    setupEventListeners() {
        const elements = {
            imageInput: document.getElementById('imageInput'),
            startWebcam: document.getElementById('startWebcam'),
            stopWebcam: document.getElementById('stopWebcam')
        };
        // Verifica se os elementos existem antes de adicionar os listeners
        for (const [id, element] of Object.entries(elements)) {
            if (!element) {
                console.warn(`Elemento ${id} não encontrado no DOM`);
                continue;
            }
            if (id === 'imageInput') element.addEventListener('change', (e) => this.handleImageUpload(e));
            if (id === 'startWebcam') element.addEventListener('click', () => this.startWebcam());
            if (id === 'stopWebcam') element.addEventListener('click', () => this.stopWebcam());
        }
    }

    async handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        const img = new Image();
        img.onload = async () => {
            await this.classifyImage(img);
        };
        img.src = URL.createObjectURL(file);
    }

    async classifyImage(imageElement) {
        // Verifica se o modelo MobileNet foi carregado
        if (!this.models.mobilenet) {
            console.error('Modelo MobileNet não carregado');
            const classificationResults = document.getElementById('classificationResults');
            if (classificationResults) {
                classificationResults.innerHTML = '<p class="error">Modelo MobileNet não carregado</p>';
            }
            return;
        }

        try {
            console.log('classifica imagem');
            const canvas = document.getElementById('imageCanvas');  //
            const ctx = canvas.getContext('2d'); //
            
            // Calcular proporção para manter aspect ratio
            const aspectRatio = imageElement.width / imageElement.height;
            let width = 300;
            let height = 300 / aspectRatio;
            if (height > 300) {
                height = 300;
                width = 300 * aspectRatio;
            }
            
            canvas.width = width;
            canvas.height = height;
            ctx.drawImage(imageElement, 0, 0, width, height);

            const predictions = await this.models.mobilenet.classify(imageElement); // Classifica a imagem
            console.log('Predições:', predictions);
            this.displayClassificationResults(predictions);
        } catch (error) {
            console.error('Erro na classificação:', error);
            const classificationResults = document.getElementById('classificationResults');
            if (classificationResults) {
                classificationResults.innerHTML = `<p class="error">Erro na classificação: ${error.message}</p>`;
            }
        }
    }

    displayClassificationResults(predictions) {
        const resultsDiv = document.getElementById('classificationResults');
        resultsDiv.innerHTML = predictions.map(prediction => `
            <div class="prediction">
                <span class="class-name">${prediction.className}</span>
                <span class="confidence">${(prediction.probability * 100).toFixed(1)}%</span>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${prediction.probability * 100}%"></div>
                </div>
            </div>
        `).join('');
    }

    async startWebcam() {
        const detectionStatus = document.getElementById('detectionStatus');
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 } 
            });
            
            const video = document.getElementById('webcam');
            video.srcObject = stream;
            this.webcamStream = stream;
            
            video.addEventListener('loadeddata', () => {
                this.startObjectDetection();
            });
            
            detectionStatus.textContent = 'detectando objetos';
        } catch (error) {
            console.error('Erro ao acessar webcam:', error);
            detectionStatus.textContent = `Erro ao acessar webcam: ${error.message}`;
        }
    }

    stopWebcam() {
        if (this.webcamStream) {
            this.webcamStream.getTracks().forEach(track => track.stop());
            this.webcamStream = null;
        }
        this.isDetecting = false;
        const detectionStatus = document.getElementById('detectionStatus');
        if (detectionStatus) {
            detectionStatus.textContent = 'Webcam parada';
        } else {
            console.warn('Elemento detectionStatus não encontrado');
        }
    }

    async startObjectDetection() {
        this.isDetecting = true;
        const video = document.getElementById('webcam');
        const canvas = document.getElementById('detectionCanvas');
        const ctx = canvas.getContext('2d');
        
        const container = document.querySelector('.video-container');
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
        
        const detectLoop = async () => {
            if (!this.isDetecting) return;
            
            try {
                const predictions = await this.models.cocoSsd.detect(video);
                
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                predictions.forEach(prediction => {
                    const [x, y, width, height] = prediction.bbox;
                    
                    ctx.strokeStyle = '#00ff00';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(x, y, width, height);
                    
                    ctx.fillStyle = '#00ff00';
                    ctx.font = '16px Arial';
                    ctx.fillText(
                        `${prediction.class} (${(prediction.score * 100).toFixed(0)}%)`,
                        x, y > 20 ? y - 5 : y + 20
                    );
                });
                
                requestAnimationFrame(detectLoop);
            } catch (error) {
                console.error('Erro na detecção:', error);
                const detectionStatus = document.getElementById('detectionStatus');
                if (detectionStatus) {
                    detectionStatus.textContent = `Erro na detecção: ${error.message}`;
                }
            }
        };
        
        detectLoop();
    }

}

function showExample(event, exampleId) {
    document.querySelectorAll('.examples-nav button').forEach(btn => 
        btn.classList.remove('active'));
    document.querySelectorAll('.example-content').forEach(content => 
        content.classList.remove('active'));
    
    event.target.classList.add('active');
    const content = document.getElementById(exampleId);
    if (content) {
        content.classList.add('active');
    } else {
        console.warn(`Elemento ${exampleId} não encontrado`);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new ComputerVisionDemo();
});