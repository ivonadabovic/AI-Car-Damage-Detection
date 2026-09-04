document.addEventListener("DOMContentLoaded", function () {

    // ============================================
    // ELEMENTI SA STRANICE
    // ============================================

    const uploadArea = document.getElementById("uploadArea");
    const imageInput = document.getElementById("imageInput");
    const previewContainer = document.getElementById("previewContainer");
    const canvas = document.getElementById("resultCanvas");
    const ctx = canvas.getContext("2d");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const clearBtn = document.getElementById("clearBtn");
    const detectionList = document.getElementById("detectionList");
    const totalDetections = document.getElementById("totalDetections");
    const damageStatus = document.getElementById("damageStatus");
    const damageCounter = document.getElementById("damageCounter");
    const resultsContainer = document.getElementById("resultsContainer");
    const recommendBtn = document.getElementById("recommendBtn");
    const drawHint = document.getElementById("drawHint");
    const dugmeProcjena = document.getElementById("dugmeProcjena");
    const dugmeUsluge = document.getElementById("dugmeUsluge");
    const removeImageBtn = document.getElementById("removeImageBtn");

    // ============================================
    // PODACI APLIKACIJE
    // ============================================

    let slika = new Image();
    let detekcije = [];
    let privremeniOkvir = null;
    let crta = false;
    let startX = 0;
    let startY = 0;
    let yoloSession = null;
    let modelUcitan = false;

    // ============================================
    // YOLO KLASE (CarDD - 6 kategorija)
    // ============================================

    const YOLO_CLASSES = [
        "dent",           // 0 - Udubljenje
        "scratch",        // 1 - Ogrebotina
        "crack",          // 2 - Pukotina
        "glass_shatter",  // 3 - Razbijeno staklo
        "lamp_broken",    // 4 - Oštećen far
        "tire_flat"       // 5 - Probušena guma
    ];

    const CLASS_NAMES = {
        "dent": "Udubljenje",
        "scratch": "Ogrebotina",
        "crack": "Pukotina",
        "glass_shatter": "Razbijeno staklo",
        "lamp_broken": "Oštećen far",
        "tire_flat": "Probušena guma"
    };

    const CLASS_COLORS = {
        "dent": "#e63946",
        "scratch": "#f77f00",
        "crack": "#d62828",
        "glass_shatter": "#6c757d",
        "lamp_broken": "#ffd60a",
        "tire_flat": "#1b4332"
    };

    const CLASS_ICONS = {
        "dent": "🔴",
        "scratch": "🔄",
        "crack": "⚡",
        "glass_shatter": "💎",
        "lamp_broken": "💡",
        "tire_flat": "⚫"
    };

    // ============================================
    // MAPA ZA PREPORUKE - SVI NA POLIRANJE
    // ============================================

    const mapaUsluga = {
        "Udubljenje": { usluga: "Poliranje", cijena: "od 300€" },
        "Ogrebotina": { usluga: "Poliranje", cijena: "od 300€" },
        "Pukotina": { usluga: "Poliranje", cijena: "od 300€" },
        "Razbijeno staklo": { usluga: "Poliranje", cijena: "od 300€" },
        "Oštećen far": { usluga: "Poliranje", cijena: "od 300€" },
        "Probušena guma": { usluga: "Poliranje", cijena: "od 300€" }
    };

    // ============================================
    // UČITAVANJE YOLO ONNX MODELA
    // ============================================

    async function ucitajYOLOModel() {
        try {
            drawHint.textContent = "📥 Učitavanje YOLO modela...";
            drawHint.style.background = "rgba(0,0,0,0.8)";

            const modelPath = 'best.onnx';
            
            try {
                const response = await fetch(modelPath);
                if (!response.ok) {
                    throw new Error('Model nije pronađen. Provjerite da li je best.onnx u root folderu.');
                }
            } catch (e) {
                console.error('❌ Greška:', e.message);
                drawHint.textContent = "❌ Model nije pronađen! Provjerite folder";
                drawHint.style.background = "rgba(211, 47, 47, 0.9)";
                return;
            }

            yoloSession = await ort.InferenceSession.create(modelPath);
            modelUcitan = true;
            console.log("✅ YOLO ONNX model učitan!");

            drawHint.textContent = "✅ YOLO model spreman - uploadujte sliku";
            drawHint.style.background = "rgba(46, 125, 50, 0.9)";

            if (slika.src && slika.complete && slika.naturalWidth > 0) {
                await pokreniYOLODetekciju();
            }

        } catch (greska) {
            console.error("❌ Greška pri učitavanju:", greska);
            drawHint.textContent = "❌ Greška: " + greska.message;
            drawHint.style.background = "rgba(211, 47, 47, 0.9)";
            modelUcitan = false;
        }
    }

    // ============================================
    // YOLO DETEKCIJA - NAJMANJI PRAG 10%
    // ============================================

    async function pokreniYOLODetekciju() {
        if (!modelUcitan || !yoloSession) {
            await ucitajYOLOModel();
            if (!modelUcitan) return;
        }

        try {
            drawHint.textContent = "🧠 YOLO analizira sliku...";
            drawHint.style.background = "rgba(0,0,0,0.8)";
            analyzeBtn.disabled = true;
            resultsContainer.style.display = "none";

            const w = slika.naturalWidth;
            const h = slika.naturalHeight;
            
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 640;
            tempCanvas.height = 640;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(slika, 0, 0, 640, 640);
            
            const imageData = tempCtx.getImageData(0, 0, 640, 640);
            const data = imageData.data;
            
            const tensor = new Float32Array(1 * 3 * 640 * 640);
            
            for (let i = 0; i < data.length / 4; i++) {
                const pixelIndex = i * 4;
                const r = data[pixelIndex] / 255.0;
                const g = data[pixelIndex + 1] / 255.0;
                const b = data[pixelIndex + 2] / 255.0;
                
                const row = Math.floor(i / 640);
                const col = i % 640;
                
                tensor[0 * 640 * 640 + row * 640 + col] = r;
                tensor[1 * 640 * 640 + row * 640 + col] = g;
                tensor[2 * 640 * 640 + row * 640 + col] = b;
            }
            
            const tensorOrt = new ort.Tensor('float32', tensor, [1, 3, 640, 640]);
            
            const feeds = { images: tensorOrt };
            const results = await yoloSession.run(feeds);
            
            // ============================================
            // OBRADA YOLO REZULTATA
            // ============================================
            
            const output = results['output0'];
            const outputData = output.data;
            const dims = output.dims;

            console.log("📊 YOLO output shape:", dims);

            const NUM_CLASSES = 6;
            const NUM_VALUES = 4 + NUM_CLASSES;

            let numDetections = 0;
            let isTransposed = false;

            if (dims.length === 3) {
                if (dims[1] === NUM_VALUES && dims[2] > 100) {
                    numDetections = dims[2];
                    isTransposed = false;
                    console.log("📊 Format: [1, " + NUM_VALUES + ", " + numDetections + "]");
                    console.log("📊 Broj klasa: " + NUM_CLASSES);
                } else if (dims[1] > 100 && dims[2] === NUM_VALUES) {
                    numDetections = dims[1];
                    isTransposed = true;
                    console.log("📊 Format: [1, " + numDetections + ", " + NUM_VALUES + "]");
                } else {
                    console.log("⚠️ Nepoznat format:", dims);
                    numDetections = Math.max(dims[1], dims[2]);
                }
            }

            const allDetections = [];
            const confidenceThreshold = 0.1;

            function getCarDDClass(classId) {
                return classId;
            }

            for (let i = 0; i < numDetections; i++) {
                let x1, y1, x2, y2;
                let maxConf = 0;
                let maxClass = 0;
                
                if (isTransposed) {
                    const idx = i * NUM_VALUES;
                    x1 = outputData[idx + 0];
                    y1 = outputData[idx + 1];
                    x2 = outputData[idx + 2];
                    y2 = outputData[idx + 3];
                    
                    for (let c = 0; c < NUM_CLASSES; c++) {
                        const conf = outputData[idx + 4 + c];
                        if (conf > maxConf) {
                            maxConf = conf;
                            maxClass = c;
                        }
                    }
                } else {
                    x1 = outputData[i * NUM_VALUES + 0];
                    y1 = outputData[i * NUM_VALUES + 1];
                    x2 = outputData[i * NUM_VALUES + 2];
                    y2 = outputData[i * NUM_VALUES + 3];
                    
                    for (let c = 0; c < NUM_CLASSES; c++) {
                        const conf = outputData[i * NUM_VALUES + 4 + c];
                        if (conf > maxConf) {
                            maxConf = conf;
                            maxClass = c;
                        }
                    }
                }
                
                if (x1 < 0 || x1 > 1 || y1 < 0 || y1 > 1 || x2 < 0 || x2 > 1 || y2 < 0 || y2 > 1) {
                    continue;
                }
                
                if (maxConf > 1) {
                    maxConf = maxConf / 100;
                }
                
                if (maxConf > confidenceThreshold && maxConf <= 1) {
                    const scaleX = w / 640;
                    const scaleY = h / 640;
                    
                    const cx = x1 * scaleX;
                    const cy = y1 * scaleY;
                    const boxW = x2 * scaleX;
                    const boxH = y2 * scaleY;
                    
                    const carDDClass = getCarDDClass(maxClass);
                    const className = YOLO_CLASSES[carDDClass] || "nepoznato";
                    
                    allDetections.push({
                        x: Math.max(0, cx - boxW/2),
                        y: Math.max(0, cy - boxH/2),
                        width: Math.min(w, boxW),
                        height: Math.min(h, boxH),
                        classId: carDDClass,
                        className: className,
                        confidence: maxConf,
                        centerY: cy / h,
                        naziv: CLASS_NAMES[className] || className
                    });
                }
            }

            console.log(`📊 Ukupno detekcija: ${allDetections.length}`);

            // ============================================
            // Non-Maximum Suppression (NMS)
            // ============================================
            
            const nmsDetections = [];
            const sorted = allDetections.sort((a, b) => b.confidence - a.confidence);

            while (sorted.length > 0) {
                const best = sorted.shift();
                nmsDetections.push(best);
                
                for (let i = sorted.length - 1; i >= 0; i--) {
                    const a = best;
                    const b = sorted[i];
                    
                    const x1 = Math.max(a.x, b.x);
                    const y1 = Math.max(a.y, b.y);
                    const x2 = Math.min(a.x + a.width, b.x + b.width);
                    const y2 = Math.min(a.y + a.height, b.y + b.height);
                    
                    if (x1 < x2 && y1 < y2) {
                        const intersection = (x2 - x1) * (y2 - y1);
                        const areaA = a.width * a.height;
                        const areaB = b.width * b.height;
                        const iou = intersection / (areaA + areaB - intersection);
                        
                        if (iou > 0.45) {
                            sorted.splice(i, 1);
                        }
                    }
                }
            }

            console.log(`📊 Nakon NMS: ${nmsDetections.length} detekcija`);

            // ============================================
            // FILTRIRANJE - NAJMANJI PRAG 10%
            // ============================================

            console.log("🔍 FILTRIRANJE - najmanji prag 10%");

            const filteredDetections = nmsDetections.filter(d => {
                const conf = d.confidence;
                const centerY = d.centerY;
                const sirina = d.width;
                const visina = d.height;
                
                // SVI PRAGOVI 10% - detektuje SVE
                if (d.className === "scratch") {
                    return conf > 0.10;
                }
                if (d.className === "dent") {
                    return conf > 0.10 && sirina > 5 && visina > 5;
                }
                if (d.className === "crack") {
                    return conf > 0.10 && centerY < 0.60 && sirina > 5 && visina > 5;
                }
                if (d.className === "tire_flat") {
                    return conf > 0.10 && centerY > 0.35;
                }
                if (d.className === "lamp_broken") {
                    return conf > 0.10 && centerY < 0.55 && sirina > 5 && visina > 5;
                }
                if (d.className === "glass_shatter") {
                    return conf > 0.10 && centerY < 0.55 && sirina > 5 && visina > 5;
                }
                return conf > 0.10;
            });

            console.log(`📊 Nakon filtera (10%): ${filteredDetections.length} detekcija`);

            // ============================================
            // DETEKCIJA KOŽE - UKLONJENA
            // ============================================

            // ============================================
            // ODABIR: SAMO YOLO -> POLIRANJE
            // ============================================
            
            let odabraniNaziv = null;
            let odabranaConf = 0;
            let odabraniBox = null;
            let odabranaKlasa = null;
            let odabranaBoja = "#e63946";
            let odabranaOznaka = "⚠️";
            let jeYoloDetekcija = false;
            let usluga = "";
            let cijena = "";

            // ============================================
            // 1. YOLO DETEKCIJE - JEDINA OPCIJA
            // ============================================
            if (filteredDetections.length > 0) {
                const najboljaYolo = filteredDetections.sort((a, b) => b.confidence - a.confidence)[0];
                odabraniNaziv = "Karoserija";
                odabranaConf = najboljaYolo.confidence;
                odabraniBox = najboljaYolo;
                odabranaKlasa = najboljaYolo.className;
                odabranaBoja = "#f77f00";
                odabranaOznaka = "🔄";
                jeYoloDetekcija = true;
                usluga = "Poliranje";
                cijena = "od 300€";
                console.log(`✅ YOLO: Poliranje (${Math.round(odabranaConf * 100)}%)`);
            }
            // ============================================
            // 2. NEMA NIČEGA
            // ============================================
            else {
                detekcije = [];
                console.log("❌ NEMA DETEKCIJA!");
                drawHint.textContent = "✅ Vozilo nema vidljivih oštećenja";
                drawHint.style.background = "rgba(46, 125, 50, 0.9)";
                analyzeBtn.disabled = false;
                nacrtajSve();
                azurirajBrojac();
                return;
            }

            // ============================================
            // KREIRANJE DETEKCIJE ZA PRIKAZ
            // ============================================
            
            if (odabraniNaziv && odabranaConf > 0.05) {
                let boxX = 0, boxY = 0, boxW = 0, boxH = 0;
                let prikaziBox = false;
                
                if (jeYoloDetekcija && odabraniBox) {
                    boxX = odabraniBox.x;
                    boxY = odabraniBox.y;
                    boxW = odabraniBox.width;
                    boxH = odabraniBox.height;
                    prikaziBox = true;
                }
                
                detekcije = [{
                    x: boxX,
                    y: boxY,
                    width: boxW,
                    height: boxH,
                    vrijednost: odabranaKlasa || "unknown",
                    naziv: odabraniNaziv,
                    boja: odabranaBoja,
                    oznaka: odabranaOznaka,
                    pouzdanost: odabranaConf,
                    yolo: jeYoloDetekcija,
                    grouped: false,
                    count: 1,
                    usluga: usluga,
                    cijena: cijena,
                    prikaziBox: prikaziBox
                }];
                
                console.log(`✅ Odabrano: ${odabraniNaziv} → ${usluga} (${Math.round(odabranaConf * 100)}%)`);
            } else {
                detekcije = [];
                console.log("ℹ️ Nema sigurnih detekcija");
            }

            if (detekcije.length > 0) {
                drawHint.textContent = `🎯 Preporuka: ${detekcije[0].usluga}`;
                drawHint.style.background = "rgba(46, 125, 50, 0.9)";
            } else {
                drawHint.textContent = "✅ Vozilo nema vidljivih oštećenja";
                drawHint.style.background = "rgba(46, 125, 50, 0.9)";
            }

            analyzeBtn.disabled = false;
            nacrtajSve();
            azurirajBrojac();

            if (detekcije.length > 0) {
                setTimeout(() => prikaziRezultate(), 300);
            }

        } catch (greska) {
            console.error("❌ Greška pri YOLO detekciji:", greska);
            drawHint.textContent = "❌ YOLO detekcija nije uspjela";
            drawHint.style.background = "rgba(211, 47, 47, 0.9)";
            analyzeBtn.disabled = false;
        }
    }

    // ============================================
    // UPLOAD SLIKE
    // ============================================

    uploadArea.addEventListener("click", function (e) {
        e.stopPropagation();
        imageInput.click();
    });

    imageInput.addEventListener("click", function (e) {
        e.stopPropagation();
    });

    imageInput.addEventListener("change", function (e) {
        if (imageInput.files.length > 0) {
            obradiSliku(imageInput.files[0]);
        }
        imageInput.value = "";
    });

    uploadArea.addEventListener("dragover", function (event) {
        event.preventDefault();
        uploadArea.classList.add("dragover");
    });

    uploadArea.addEventListener("dragleave", function () {
        uploadArea.classList.remove("dragover");
    });

    uploadArea.addEventListener("drop", function (event) {
        event.preventDefault();
        uploadArea.classList.remove("dragover");
        const fajl = event.dataTransfer.files[0];
        if (fajl) obradiSliku(fajl);
    });

    // ============================================
    // OBRADA SLIKE
    // ============================================

    function obradiSliku(fajl) {
        if (!fajl.type.startsWith("image/")) {
            alert("Molimo vas odaberite fotografiju.");
            return;
        }

        const reader = new FileReader();
        reader.onload = function (event) {
            slika.onload = async function () {
                prikaziSliku();
                if (modelUcitan) {
                    await pokreniYOLODetekciju();
                } else {
                    drawHint.textContent = "⏳ Učitavanje YOLO modela...";
                    await ucitajYOLOModel();
                }
            };
            slika.src = event.target.result;
        };
        reader.readAsDataURL(fajl);
    }

    // ============================================
    // PRIKAZ SLIKE NA CANVASU
    // ============================================

    function prikaziSliku() {
        previewContainer.style.display = "block";
        uploadArea.classList.add("has-image");

        const maxWidth = 800;
        const maxHeight = 500;
        let w = slika.naturalWidth;
        let h = slika.naturalHeight;
        
        if (w > maxWidth || h > maxHeight) {
            const scale = Math.min(maxWidth / w, maxHeight / h);
            w = Math.floor(w * scale);
            h = Math.floor(h * scale);
        }
        
        canvas.width = w;
        canvas.height = h;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(slika, 0, 0, canvas.width, canvas.height);

        resultsContainer.style.display = "none";
        detekcije = [];
        azurirajBrojac();
        napraviAlate();
    }

    // ============================================
    // CRTANJE
    // ============================================

    function nacrtajSve() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(slika, 0, 0, canvas.width, canvas.height);

        detekcije.forEach(function (detekcija) {
            if (!detekcija.prikaziBox) {
                const labela = `${detekcija.oznaka || '⚠️'} ${detekcija.naziv} → ${detekcija.usluga} (${Math.round(detekcija.pouzdanost * 100)}%)`;
                
                const padding = 8;
                ctx.font = "bold 13px Inter";
                const metrics = ctx.measureText(labela);
                const labelW = metrics.width + padding * 2;
                const labelH = 28;
                const labelX = 10;
                const labelY = 10;

                ctx.shadowColor = "rgba(0,0,0,0.15)";
                ctx.shadowBlur = 8;
                ctx.fillStyle = "rgba(255,255,255,0.93)";
                ctx.beginPath();
                ctx.roundRect(labelX, labelY, labelW, labelH, 8);
                ctx.fill();
                ctx.shadowBlur = 0;

                ctx.fillStyle = detekcija.boja || "#e63946";
                ctx.font = "bold 13px Inter";
                ctx.fillText(labela, labelX + padding, labelY + labelH - 8);
                return;
            }
            
            const boja = detekcija.boja || "#e63946";
            
            ctx.shadowColor = boja + "60";
            ctx.shadowBlur = 15;
            ctx.strokeStyle = boja;
            ctx.lineWidth = 3;
            ctx.strokeRect(detekcija.x, detekcija.y, detekcija.width, detekcija.height);
            ctx.shadowBlur = 0;

            ctx.fillStyle = boja + "15";
            ctx.fillRect(detekcija.x, detekcija.y, detekcija.width, detekcija.height);

            let labela = `${detekcija.oznaka || '⚠️'} ${detekcija.naziv}`;
            if (detekcija.usluga && detekcija.usluga !== "Pregled") {
                labela = `${detekcija.oznaka || '⚠️'} ${detekcija.naziv} → ${detekcija.usluga}`;
            }
            
            if (detekcija.pouzdanost) {
                labela += ` (${Math.round(detekcija.pouzdanost * 100)}%)`;
            }
            
            const padding = 8;
            ctx.font = "bold 13px Inter";
            const metrics = ctx.measureText(labela);
            const labelW = metrics.width + padding * 2;
            const labelH = 28;
            
            let labelX = detekcija.x;
            let labelY = detekcija.y - labelH - 6;
            if (labelY < 4) labelY = detekcija.y + 4;

            ctx.shadowColor = "rgba(0,0,0,0.15)";
            ctx.shadowBlur = 8;
            ctx.fillStyle = "rgba(255,255,255,0.93)";
            ctx.beginPath();
            ctx.roundRect(labelX, labelY, labelW, labelH, 8);
            ctx.fill();
            ctx.shadowBlur = 0;

            ctx.fillStyle = boja;
            ctx.font = "bold 13px Inter";
            ctx.fillText(labela, labelX + padding, labelY + labelH - 8);
        });

        if (privremeniOkvir) {
            ctx.strokeStyle = "#111111";
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 4]);
            ctx.strokeRect(
                privremeniOkvir.x,
                privremeniOkvir.y,
                privremeniOkvir.width,
                privremeniOkvir.height
            );
            ctx.setLineDash([]);
        }
    }

    // ============================================
    // roundRect
    // ============================================

    CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {
        if (w < 2 * r) r = w / 2;
        if (h < 2 * r) r = h / 2;
        this.moveTo(x + r, y);
        this.lineTo(x + w - r, y);
        this.quadraticCurveTo(x + w, y, x + w, y + r);
        this.lineTo(x + w, y + h - r);
        this.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        this.lineTo(x + r, y + h);
        this.quadraticCurveTo(x, y + h, x, y + h - r);
        this.lineTo(x, y + r);
        this.quadraticCurveTo(x, y, x + r, y);
        this.closePath();
        return this;
    };

    // ============================================
    // KOORDINATE
    // ============================================

    function koordinate(event) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: (event.clientX - rect.left) * scaleX,
            y: (event.clientY - rect.top) * scaleY
        };
    }

    // ============================================
    // RUČNO CRTANJE
    // ============================================

    canvas.onmousedown = function (event) {
        if (!slika.src) return;
        const pozicija = koordinate(event);
        startX = pozicija.x;
        startY = pozicija.y;
        crta = true;
        privremeniOkvir = { x: startX, y: startY, width: 0, height: 0 };
    };

    canvas.onmousemove = function (event) {
        if (!crta) return;
        const pozicija = koordinate(event);
        privremeniOkvir = {
            x: Math.min(startX, pozicija.x),
            y: Math.min(startY, pozicija.y),
            width: Math.abs(pozicija.x - startX),
            height: Math.abs(pozicija.y - startY)
        };
        nacrtajSve();
    };

    canvas.onmouseup = function () {
        if (!crta) return;
        crta = false;
        if (privremeniOkvir && privremeniOkvir.width > 10 && privremeniOkvir.height > 10) {
            otvoriIzborOstecenja(privremeniOkvir);
        }
        privremeniOkvir = null;
        nacrtajSve();
    };

    canvas.ontouchstart = function (event) {
        event.preventDefault();
        const touch = event.touches[0];
        const mouseEvent = new MouseEvent("mousedown", {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.onmousedown(mouseEvent);
    };

    canvas.ontouchmove = function (event) {
        event.preventDefault();
        const touch = event.touches[0];
        const mouseEvent = new MouseEvent("mousemove", {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.onmousemove(mouseEvent);
    };

    canvas.ontouchend = function (event) {
        event.preventDefault();
        canvas.onmouseup(event);
    };

    // ============================================
    // RUČNI IZBOR VRSTE OŠTEĆENJA
    // ============================================

    function otvoriIzborOstecenja(okvir) {
        const sveVrste = [
            { vrijednost: "karoserija", naziv: "Karoserija", boja: "#f77f00", oznaka: "🔄" },
            { vrijednost: "koza", naziv: "Koža", boja: "#9c6644", oznaka: "🧵" }
        ];

        const izbor = prompt(
            "🛠️ ODABERITE ŠTA JE NA SLICI:\n\n" +
            "1 - Karoserija → Poliranje\n" +
            "2 - Koža → Restauracija kože"
        );

        const broj = Number(izbor);
        if (!broj || broj < 1 || broj > sveVrste.length) {
            return;
        }

        const odabrano = sveVrste[broj - 1];
        
        let usluga = odabrano.vrijednost === "karoserija" ? "Poliranje" : "Restauracija kože";
        let cijena = odabrano.vrijednost === "karoserija" ? "od 300€" : "od 100€";
        
        detekcije.push({
            x: okvir.x,
            y: okvir.y,
            width: okvir.width,
            height: okvir.height,
            vrijednost: odabrano.vrijednost,
            naziv: odabrano.naziv,
            boja: odabrano.boja,
            oznaka: odabrano.oznaka,
            pouzdanost: 1.0,
            rucno: true,
            usluga: usluga,
            cijena: cijena,
            prikaziBox: true
        });

        azurirajBrojac();
        nacrtajSve();
        analyzeBtn.disabled = false;
    }

    // ============================================
    // ALATI
    // ============================================

    function napraviAlate() {
        let stariAlati = document.getElementById("damageTools");
        if (stariAlati) stariAlati.remove();

        const alati = document.createElement("div");
        alati.id = "damageTools";
        alati.style.display = "flex";
        alati.style.justifyContent = "center";
        alati.style.gap = "10px";
        alati.style.marginTop = "12px";
        alati.style.flexWrap = "wrap";
        alati.style.alignItems = "center";

        const info = document.createElement("span");
        info.textContent = "🖱️ Kliknite i prevucite za ručno označavanje";
        info.style.color = "#4a4a6a";
        info.style.fontSize = "0.85rem";
        info.style.background = "#f8f9fa";
        info.style.padding = "6px 16px";
        info.style.borderRadius = "20px";
        alati.appendChild(info);

        const reanalyzeBtn = document.createElement("button");
        reanalyzeBtn.textContent = "🔄 Ponovi YOLO analizu";
        reanalyzeBtn.style.padding = "6px 18px";
        reanalyzeBtn.style.borderRadius = "20px";
        reanalyzeBtn.style.background = "#e9ecef";
        reanalyzeBtn.style.color = "#212529";
        reanalyzeBtn.style.fontSize = "0.8rem";
        reanalyzeBtn.style.fontWeight = "600";
        reanalyzeBtn.style.border = "none";
        reanalyzeBtn.style.cursor = "pointer";
        reanalyzeBtn.style.transition = "0.3s ease";
        
        reanalyzeBtn.onmouseover = function() {
            this.style.background = "#dee2e6";
        };
        reanalyzeBtn.onmouseout = function() {
            this.style.background = "#e9ecef";
        };
        
        reanalyzeBtn.onclick = async function() {
            if (!slika.src) return;
            const rucne = detekcije.filter(d => d.rucno === true);
            detekcije = rucne;
            azurirajBrojac();
            resultsContainer.style.display = "none";
            await pokreniYOLODetekciju();
        };
        
        alati.appendChild(reanalyzeBtn);

        canvas.parentElement.parentElement.appendChild(alati);
    }

    // ============================================
    // ANALIZA
    // ============================================

    analyzeBtn.addEventListener("click", function () {
        if (detekcije.length === 0) {
            alert("✅ Nema detektovanih oštećenja na ovoj slici.");
            return;
        }
        prikaziRezultate();
    });

    // ============================================
    // PRIKAZ REZULTATA
    // ============================================

    function prikaziRezultate() {
        resultsContainer.style.display = "block";
        detectionList.innerHTML = "";

        detekcije.forEach(function (detekcija, index) {
            const element = document.createElement("div");
            element.className = "detection-item";
            const oznaka = detekcija.oznaka || "⚠️";
            const pouzdanost = detekcija.pouzdanost ? Math.round(detekcija.pouzdanost * 100) + "%" : "100%";
            
            let prikaz = `${oznaka} ${detekcija.naziv}`;
            if (detekcija.usluga && detekcija.usluga !== "Pregled") {
                prikaz += ` → ${detekcija.usluga}`;
            }
            if (detekcija.cijena && detekcija.cijena !== "po dogovoru") {
                prikaz += ` (${detekcija.cijena})`;
            }
            
            element.innerHTML = `
                <span>${prikaz}</span>
                <strong style="color:${detekcija.boja || '#e63946'}">${pouzdanost}</strong>
            `;
            detectionList.appendChild(element);
        });

        totalDetections.textContent = detekcije.length;
        
        const tezina = detekcije.length >= 4 ? "🔴 Opsežna oštećenja" :
                       detekcije.length >= 2 ? "🟡 Umjerena oštećenja" :
                       detekcije.length >= 1 ? "🟢 Laka oštećenja" : "⚪ Bez oštećenja";
        
        damageStatus.textContent = tezina;
        damageStatus.className = detekcije.length >= 3 ? "status-damaged" : "";

        if (detekcije.length > 0 && detekcije[0].usluga && detekcije[0].usluga !== "Pregled") {
            recommendBtn.textContent = `📋 Preporučena usluga: ${detekcije[0].usluga} (${detekcije[0].cijena || "po dogovoru"})`;
            recommendBtn.style.display = "inline-flex";
        } else {
            recommendBtn.textContent = "✅ Vozilo nema vidljivih oštećenja";
            recommendBtn.style.display = "inline-flex";
        }
        
        resultsContainer.scrollIntoView({ behavior: "smooth", block: "center" });
    }

    // ============================================
    // BROJAČ
    // ============================================

    function azurirajBrojac() {
        damageCounter.textContent = `🔍 Oštećenja: ${detekcije.length}`;
    }

    // ============================================
    // BRISANJE
    // ============================================

    clearBtn.addEventListener("click", ukloniSliku);
    removeImageBtn.addEventListener("click", ukloniSliku);

    function ukloniSliku() {
        slika = new Image();
        detekcije = [];
        privremeniOkvir = null;
        crta = false;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        canvas.width = 0;
        canvas.height = 0;

        uploadArea.classList.remove("has-image");
        previewContainer.style.display = "none";
        resultsContainer.style.display = "none";
        analyzeBtn.disabled = true;

        drawHint.textContent = "📷 Uploadujte sliku za YOLO analizu";
        drawHint.style.background = "rgba(0,0,0,0.7)";
        azurirajBrojac();

        const alati = document.getElementById("damageTools");
        if (alati) alati.remove();
    }

    // ============================================
    // NAVIGACIJA
    // ============================================

    dugmeProcjena.addEventListener("click", function () {
        document.getElementById("estimate").scrollIntoView({ behavior: "smooth" });
    });

    dugmeUsluge.addEventListener("click", function () {
        document.getElementById("services").scrollIntoView({ behavior: "smooth" });
    });

    // ============================================
    // PROGRESS BAR
    // ============================================

    window.addEventListener("scroll", function () {
        const visina = document.documentElement.scrollHeight - document.documentElement.clientHeight;
        const procenat = (window.scrollY / visina) * 100;
        document.querySelector(".progress-bar").style.width = procenat + "%";
    });

    // ============================================
    // INICIJALIZACIJA
    // ============================================

    console.log("🚗 TDC - YOLO AI Procjena oštećenja vozila");
    console.log("🎯 Poliranje - najmanji prag 10%");
    console.log("🔍 Detektuje sve što YOLO vidi");

    analyzeBtn.disabled = true;

    ucitajYOLOModel();

});