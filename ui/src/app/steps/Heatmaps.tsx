import { useState, useEffect } from "react";

import styles from "./heatmaps.module.css";

// COMPONENTS
import Button from "../../components/button/Button";

const UploadVideos = ({ video, onChangeStep }: { video: string, onChangeStep?: (data: any) => void } ) => {
    const [videoURL, setVideoURL] = useState<string | null>(null);
    const [imageURL, setImageURL] = useState<string | null>(null);
    const [loading, setLoading] = useState<boolean>(false);

    useEffect(() => {
        const socket = new WebSocket('ws://localhost:8765/heatmaps');

        socket.addEventListener('open', (event) => {
            console.log("Connected to WebSocket server");
            setLoading(true);
            socket.send(video);
        });

        socket.addEventListener('message', (event) => {
            const data = JSON.parse(event.data);

            if(data?.length === 2) {
                setVideoURL(data[0]);
                setImageURL(data[1]);
            }
        });

        socket.addEventListener('error', (event) => {
            console.error('WebSocket error:', event);
        });

        socket.addEventListener('close', () => {
            console.log('WebSocket connection closed');
        });
    }, []);

    useEffect(() => {
        if(videoURL && imageURL) setLoading(false);
    }, [videoURL, imageURL]);

    return (
        <div className={styles.contentContainer}>
            <div className={styles.dimensionsContainer}>
                <h2 className={styles.dimensionTitle}>DIMENSIONES</h2>
                <Button
                    text="Flujo de Personas"
                    color="black"
                    className={styles.dimensionButton}
                    disabled={loading}
                />
                <Button
                    text="Acciones"
                    color="black"
                    className={styles.dimensionButton}
                    disabled
                />
                <Button
                    text="Edades"
                    color="black"
                    className={styles.dimensionButton}
                    disabled
                />
                <Button
                    text="Género"
                    color="black"
                    className={styles.dimensionButton}
                    disabled
                />
            </div>

            {
                videoURL && imageURL && (
                    <div className={styles.videoContainer}>
                        <video className={styles.video} src={videoURL} controls autoPlay={true} height="auto" />
                    </div>
                )
            }

            <Button
                text={loading ? "Cargando..." : "Continuar"}
                color="black"
                className={styles.continueButton}
                onClick={() => {}}
                disabled={loading}
            />
        </div>
    )
}

export default UploadVideos
