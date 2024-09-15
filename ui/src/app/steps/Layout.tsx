import { useEffect, useRef, useState } from 'react'
import { S3 } from 'aws-sdk';

// COMPEONENTS
import Button from '../../components/button/Button'

import styles from './layout.module.css'

interface Point {
  x: number
  y: number
}

interface Shape {
  corners: Point[],
  colors: (string | CanvasGradient | CanvasPattern)[]
}

interface DrawState {
  isDrawing: boolean
  startX: number
  startY: number
}

const HANDLE_RADIUS = 4

export default function Layout({ video, onChangeStep }: { video: string; onChangeStep: (data: any) => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement>(null);

  const [loading, setLoading] = useState<boolean>(false);
  const [color, setColor] = useState<string[]>([]);
  const [sections, setSections] = useState<{name: string, color: [string, string]}[]>([]);
  
  const [shapes, setShapes] = useState<Shape[]>([]);
  const [draggedCorner, setDraggedCorner] = useState<{ shapeIndex: number; cornerIndex: number } | null>(null);
  const [drawState, setDrawState] = useState<DrawState>({
    isDrawing: false,
    startX: 0,
    startY: 0,
  });

  useEffect(() => {
    const canvas = canvasRef.current
    const previewCanvas = previewCanvasRef.current
    if (!canvas || !previewCanvas) return

    const ctx = canvas.getContext('2d')
    const previewCtx = previewCanvas.getContext('2d')
    if (!ctx || !previewCtx) return

    // const image = new Image()
    // image.src = 'https://www.avtech.com.tw/upload/Case/C201409001_C1.jpg' // Replace with your actual image URL

    const resizeCanvas = () => {
        canvas.width = window.innerWidth * 0.4
        canvas.height = window.innerHeight * 0.47
        previewCanvas.width = window.innerWidth * 0.4
        previewCanvas.height = window.innerHeight * 0.47
        drawAll()
    }

    const vid = document.createElement('video');
    vid.src = video.replace('hackmty.6619572a94d5a1d039dc239d940199ed.r2.cloudflarestorage.com', 'pub-f9ef82ae3ee74240886857c6bf5f4495.r2.dev').replace('6619572a94d5a1d039dc239d940199ed.r2.cloudflarestorage.com/hackmty', 'pub-f9ef82ae3ee74240886857c6bf5f4495.r2.dev');
    vid.crossOrigin = 'anonymous';
    vid.muted = true;
    vid.loop = true;
    vid.setAttribute('type', 'video/mp4');
    vid.play();

    const drawAll = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (vid.readyState >= 1) {
        console.log('ready');
        // ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        ctx.drawImage(vid, 0, 0, window.innerWidth * 0.4, window.innerHeight * 0.47);
      } else {
        console.log('not ready');
        // image.onload = () => ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        vid.onloadeddata = () => {
            if (ctx) {
                ctx.drawImage(vid, 0, 0, window.innerWidth * 0.4, window.innerHeight * 0.47);
                vid.pause();
                shapes.forEach((shape, index) => {
                    drawShape(ctx, shape)
                    drawHandles(ctx, shape, index)
                  })
            }
        }
      }
      
    }

    const drawShape = (context: CanvasRenderingContext2D, shape: Shape) => {
      context.fillStyle = shape?.colors[1]
      context.strokeStyle = shape?.colors[0]
      context.lineWidth = 2
      context.beginPath()
      context.moveTo(shape.corners[0].x, shape.corners[0].y)
      for (let i = 1; i < shape.corners.length; i++) {
        context.lineTo(shape.corners[i].x, shape.corners[i].y)
      }
      context.closePath()
      context.fill()
    }

    const drawHandles = (context: CanvasRenderingContext2D, shape: Shape, shapeIndex: number) => {
      shape.corners.forEach((corner, cornerIndex) => {
        context.fillStyle = shape.colors[0]
        context.beginPath()
        context.arc(corner.x, corner.y, HANDLE_RADIUS, 0, Math.PI * 2)
        context.fill()
        context.stroke()
      })
    }

    window.addEventListener('resize', resizeCanvas)
    
    resizeCanvas();

    return () => {
      window.removeEventListener('resize', resizeCanvas)
    }
  }, [shapes]);

  const handleMouseDown = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if(!color || color.length === 0) return;

    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top

    // Check if we're clicking on a handle
    for (let shapeIndex = 0; shapeIndex < shapes.length; shapeIndex++) {
      const shape = shapes[shapeIndex]
      for (let cornerIndex = 0; cornerIndex < shape.corners.length; cornerIndex++) {
        const corner = shape.corners[cornerIndex]
        const distance = Math.sqrt((x - corner.x) ** 2 + (y - corner.y) ** 2)
        if (distance <= HANDLE_RADIUS+2) {
          setDraggedCorner({ shapeIndex, cornerIndex })
          return
        }
      }
    }

    // If not clicking on a handle, start drawing a new shape
    setDrawState({
      isDrawing: true,
      startX: x,
      startY: y,
    })
  }

  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if(!color || color.length === 0) return;

    const canvas = canvasRef.current
    const previewCanvas = previewCanvasRef.current
    if (!canvas || !previewCanvas) return

    const rect = canvas.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top

    if (draggedCorner) {
      // Move the dragged corner
      const newShapes = [...shapes]
      newShapes[draggedCorner.shapeIndex].corners[draggedCorner.cornerIndex] = { x, y }
      setShapes(newShapes)
    } else if (drawState.isDrawing) {
      // Draw preview of new shape
      const previewCtx = previewCanvas.getContext('2d')
      if (!previewCtx) return

      previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height)
      previewCtx.fillStyle = color[1]
      previewCtx.beginPath()
      previewCtx.moveTo(drawState.startX, drawState.startY)
      previewCtx.lineTo(x, drawState.startY)
      previewCtx.lineTo(x, y)
      previewCtx.lineTo(drawState.startX, y)
      previewCtx.closePath()
      previewCtx.fill()
    }
  }

  const handleMouseUp = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if(!color || color.length === 0) return;
    
    if (draggedCorner) {
      setDraggedCorner(null)
    } else if (drawState.isDrawing) {
      const canvas = canvasRef.current
      const previewCanvas = previewCanvasRef.current
      if (!canvas || !previewCanvas) return

      const rect = canvas.getBoundingClientRect()
      const endX = event.clientX - rect.left
      const endY = event.clientY - rect.top

      const newShape: Shape = {
        corners: [
          { x: drawState.startX, y: drawState.startY },
          { x: endX, y: drawState.startY },
          { x: endX, y: endY },
          { x: drawState.startX, y: endY },
        ],
        colors: [color[0], color[1]],
      }

      setShapes(prevShapes => [...prevShapes, newShape]);

      setDrawState({
        isDrawing: false,
        startX: 0,
        startY: 0,
      })

      const previewCtx = previewCanvas.getContext('2d')
      if (previewCtx) {
        previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height)
      }
    }
  }

  const addSection = () => {
    const [r, g, b] = [Math.floor(Math.random() * 256), Math.floor(Math.random() * 256), Math.floor(Math.random() * 256)];
    const newColors = [
        `rgba(${r}, ${g}, ${b}, 1)`,
        `rgba(${r}, ${g}, ${b}, 0.5)`
    ];

    if (sections.some(section => section.color[0] === newColors[0])) {
      addSection();
      return;
    }

    setColor(newColors);
    setSections(prevSections => [...prevSections, { name: `Sección ${prevSections.length + 1}`, color: [newColors[0], newColors[1]] }]);
  }

  const removeSection = (index: number) => {
    setSections(prevSections => prevSections.filter((_, i) => i !== index));
    setShapes(prevShapes => prevShapes.filter(
        shape => shape.colors[0] !== sections[index].color[0] && shape.colors[1] !== sections[index].color[1]
    ));

    if (color[0] === sections[index].color[0]) {
        setColor([]);
    }
  }

const uploadLayout = async () => {
    setLoading(true);

    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.toBlob(async (blob) => {
        if (!blob) return;

        const s3 = new S3({
            endpoint: 'https://6619572a94d5a1d039dc239d940199ed.r2.cloudflarestorage.com',
            accessKeyId: 'b19d514f16f18783ff2a290c04f619be',
            secretAccessKey: 'bd6a46cbdf3fd63f718caba171cd8fd5ecd917db709246499dc60371b935d31d',
            region: 'auto',
            signatureVersion: 'v4',
        });

        const params = {
            Bucket: 'hackmty',
            Key: `${Date.now()}_layout.png`,
            Body: blob,
            ContentType: 'image/png',
        };

        try {
            const data = await s3.upload(params).promise();
            onChangeStep({ layout: data.Location });
            return data.Location;
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    }, 'image/png');
}

  return (
    <div className={styles.contentContainer}>
        <div className={styles.sectionsGlobalContainer}>
            <h1 className={styles.title}>SECCIONES</h1>
            <p className={styles.description}>
                Añade secciones y dibuja en la imagen para dividir el espacio en diferentes áreas.
            </p>
            <div className={styles.sectionsContainer}>
                {
                    sections.map((section, index) => {
                        const active = color[0] === section.color[0];

                        return (
                            <div
                                className={`${styles.sectionContainer} ${active && styles.activeSection}`}
                                onClick={() => setColor(section.color)}
                            >
                                <div className={styles.colorAndTextContainer}>
                                    
                                    <div
                                        className={`${styles.colorContainer} ${!active && styles.inactiveColorContainer}`}
                                        style={{ backgroundColor: section.color[0] }}
                                    >
                                        <div
                                            className={styles.color}
                                            style={{
                                                backgroundColor: section.color[0],
                                                border: `3px solid ${!active ? section.color[0] : "var(--white)"}`
                                            }}
                                        ></div>
                                    </div>
                                    
                                    <input
                                        value={section.name}
                                        onChange={(event) => {
                                            const newSections = [...sections];
                                            newSections[index].name = event.target.value;
                                            setSections(newSections);
                                        }}
                                        className={styles.sectionName}
                                    />
                                    {/* <p className={styles.sectionName}>{section.name}</p> */}
                                </div>
                                <svg
                                    width="21"
                                    height="22"
                                    viewBox="0 0 21 22"
                                    fill="none"
                                    xmlns="http://www.w3.org/2000/svg"
                                    className={styles.removeSectionIcon}
                                    onClick={(event) => {
                                        event.stopPropagation();
                                        removeSection(index);
                                    }}
                                >
                                    <path d="M7.21875 3.125H13.7812C13.9553 3.125 14.1222 3.05586 14.2453 2.93279C14.3684 2.80972 14.4375 2.6428 14.4375 2.46875C14.4375 2.2947 14.3684 2.12778 14.2453 2.00471C14.1222 1.88164 13.9553 1.8125 13.7812 1.8125H7.21875C7.0447 1.8125 6.87778 1.88164 6.75471 2.00471C6.63164 2.12778 6.5625 2.2947 6.5625 2.46875C6.5625 2.6428 6.63164 2.80972 6.75471 2.93279C6.87778 3.05586 7.0447 3.125 7.21875 3.125Z" fill="black"/>
                                    <path d="M17.7188 4.4375H3.28125C3.1072 4.4375 2.94028 4.50664 2.81721 4.62971C2.69414 4.75278 2.625 4.9197 2.625 5.09375C2.625 5.2678 2.69414 5.43472 2.81721 5.55779C2.94028 5.68086 3.1072 5.75 3.28125 5.75H3.9375V17.5625C3.93784 17.9105 4.07623 18.2441 4.3223 18.4902C4.56837 18.7363 4.90201 18.8747 5.25 18.875H15.75C16.098 18.8747 16.4316 18.7363 16.6777 18.4902C16.9238 18.2441 17.0622 17.9105 17.0625 17.5625V5.75H17.7188C17.8928 5.75 18.0597 5.68086 18.1828 5.55779C18.3059 5.43472 18.375 5.2678 18.375 5.09375C18.375 4.9197 18.3059 4.75278 18.1828 4.62971C18.0597 4.50664 17.8928 4.4375 17.7188 4.4375ZM9.1875 14.2812C9.1875 14.4553 9.11836 14.6222 8.99529 14.7453C8.87222 14.8684 8.7053 14.9375 8.53125 14.9375C8.3572 14.9375 8.19028 14.8684 8.06721 14.7453C7.94414 14.6222 7.875 14.4553 7.875 14.2812V9.03125C7.875 8.8572 7.94414 8.69028 8.06721 8.56721C8.19028 8.44414 8.3572 8.375 8.53125 8.375C8.7053 8.375 8.87222 8.44414 8.99529 8.56721C9.11836 8.69028 9.1875 8.8572 9.1875 9.03125V14.2812ZM13.125 14.2812C13.125 14.4553 13.0559 14.6222 12.9328 14.7453C12.8097 14.8684 12.6428 14.9375 12.4688 14.9375C12.2947 14.9375 12.1278 14.8684 12.0047 14.7453C11.8816 14.6222 11.8125 14.4553 11.8125 14.2812V9.03125C11.8125 8.8572 11.8816 8.69028 12.0047 8.56721C12.1278 8.44414 12.2947 8.375 12.4688 8.375C12.6428 8.375 12.8097 8.44414 12.9328 8.56721C13.0559 8.69028 13.125 8.8572 13.125 9.03125V14.2812Z" fill="black"/>
                                </svg>
                            </div>
                        )
                    })
                }
                <Button
                    text="Añadir Sección"
                    color='black'
                    className={`${styles.addSectionButton} ${sections.length > 0 && styles.addSectionButtonWithSections}`}
                    onClick={() => addSection()}
                >
                    <svg width="12" height="12" viewBox="0 0 28 21" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M19.6123 15.5673H5.69403V1.64892C5.69403 1.42844 5.60644 1.217 5.45054 1.0611C5.29465 0.905202 5.0832 0.817619 4.86273 0.817619C4.64226 0.817619 4.43081 0.905201 4.27491 1.0611C4.11902 1.217 4.03143 1.42844 4.03143 1.64892V4.02405H1.6563C1.43583 4.02405 1.22438 4.11163 1.06849 4.26753C0.912587 4.42343 0.825004 4.63487 0.825004 4.85534C0.825004 5.07582 0.912587 5.28726 1.06849 5.44316C1.22438 5.59906 1.43583 5.68664 1.6563 5.68664H4.03143V16.3986C4.03144 16.619 4.11903 16.8305 4.27492 16.9864C4.43082 17.1423 4.64225 17.2299 4.86272 17.2299H15.5746V19.605C15.5746 19.8255 15.6622 20.0369 15.8181 20.1928C15.974 20.3487 16.1854 20.4363 16.4059 20.4363C16.6263 20.4363 16.8378 20.3487 16.9937 20.1928C17.1496 20.0369 17.2372 19.8255 17.2372 19.605V17.2299H19.6123C19.8328 17.2299 20.0442 17.1423 20.2001 16.9864C20.356 16.8305 20.4436 16.619 20.4436 16.3986C20.4436 16.1781 20.356 15.9667 20.2001 15.8108C20.0442 15.6549 19.8328 15.5673 19.6123 15.5673Z" fill="white" stroke="white" stroke-width="0.380021"/>
                        <path d="M8.18718 4.92659H16.3346V13.0734C16.3346 13.2939 16.4222 13.5053 16.5781 13.6612C16.734 13.8171 16.9454 13.9047 17.1659 13.9047C17.3864 13.9047 17.5978 13.8171 17.7537 13.6612C17.9096 13.5053 17.9972 13.2939 17.9972 13.0734V4.0953C17.9972 3.87482 17.9096 3.66338 17.7537 3.50748C17.5978 3.35158 17.3864 3.264 17.1659 3.264H8.18718C7.96671 3.264 7.75527 3.35158 7.59937 3.50748C7.44347 3.66338 7.35589 3.87482 7.35589 4.0953C7.35589 4.31577 7.44347 4.52721 7.59937 4.68311C7.75527 4.83901 7.96671 4.92659 8.18718 4.92659Z" fill="white" stroke="white" stroke-width="0.380021"/>
                    </svg>
                </Button>
            </div>
        </div>
        <div className={styles.canvasContainer}>
            <canvas
                ref={canvasRef}
                className={styles.canvas}
            />
            <canvas
                ref={previewCanvasRef}
                className={styles.canvas}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
            />
        </div>
        <Button
            text={loading ? "Cargando..." : "Continuar"}
            color="black"
            className={styles.continueButton}
            onClick={() => uploadLayout()}
            disabled={loading || sections.length === 0}
        />
    </div>
  )
}
