import { useState, useRef, useEffect } from "react";
import { S3 } from 'aws-sdk';

import styles from "./uploadVideos.module.css";

// COMPONENTS
import Button from "../../components/button/Button";

const UploadVideos = ({ onChangeStep }: { onChangeStep?: (data: any) => void } ) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [videos, setVideos] = useState<File[]>([]);
  const [videoURLs, setVideoURLs] = useState<(string|undefined)[]>([]);
  const [loading, setLoading] = useState(false);

  const handleUpload = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    const filesArray = Array.from(files || []);

    if(filesArray.length > 8 || videos.length + filesArray.length > 8) {
        alert("El número máximo de archivos permitidos es 8");
        return;
    }

    if(filesArray.some(file => file.size > 200000000)) {
        alert("El tamaño máximo de archivo permitido es 200 MB");
        return;
    }

    if(filesArray.some(file => file.type != "video/mp4")) {
        alert("El archivo seleccionado no es un video");
        return;
    }

    if (files) setVideos([...videos, ...Array.from(files)]);
  };

  const handleDelete = (index: number) => {
    const newVideos = videos.filter((_, i) => i !== index);
    setVideos(newVideos);
  };

  const generateThumbnail = (video: File) => {
    return URL.createObjectURL(video);
  }

  const UploadVideos = async () => {
    setLoading(true);

    const s3 = new S3({
        endpoint: 'https://6619572a94d5a1d039dc239d940199ed.r2.cloudflarestorage.com',
        accessKeyId: 'b19d514f16f18783ff2a290c04f619be',
        secretAccessKey: 'bd6a46cbdf3fd63f718caba171cd8fd5ecd917db709246499dc60371b935d31d',
        region: 'auto',
        signatureVersion: 'v4',
    });

    await Promise.all(videos.map(async (video, index) => {
        const params = {
            Bucket: 'hackmty',
            Key: `${Date.now()}_${video.name}`,
            Body: video,
            ContentType: video.type,
        };

        try {
            const data = await s3.upload(params).promise();
            return data.Location;
        } catch (error) {
            console.error(error);
        }
    })).then((urls) => {
        setVideoURLs(urls);
    });

    setLoading(false);
  }

  useEffect(() => {
    if(videoURLs.length > 0) {
        console.log(videoURLs);
        onChangeStep && onChangeStep({videos: videoURLs});
    }
  }, [videoURLs]);

  return (
    <div className={styles.contentContainer}>
        <input
            ref={fileInputRef}
            type="file"
            accept=".mp4"
            multiple
            onChange={handleFileChange}
            style={{display: "none"}}
        />
        {
            videos.length == 0 && (
                <div className={styles.uploadButtonContainer}>
                    <Button
                        text="Seleccionar archivos"
                        color="black"
                        className={styles.uploadButton}
                        onClick={handleUpload}
                    >
                        <svg className={styles.fileIcon} width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M30.235 13.8306C30.0495 13.5738 29.8057 13.3646 29.5236 13.2202C29.2416 13.0759 28.9294 13.0004 28.6125 13H27V11C26.9994 10.4698 26.7885 9.9614 26.4135 9.58646C26.0386 9.21151 25.5302 9.00061 25 9H16.3334L12.8667 6.40002C12.5201 6.14119 12.0992 6.00092 11.6666 6H5C4.46975 6.00061 3.9614 6.21151 3.58646 6.58646C3.21151 6.9614 3.00061 7.46975 3 8V26C3.00004 26.0353 3.00193 26.0706 3.00568 26.1057C3.00604 26.1091 3.00684 26.1125 3.00726 26.1159C3.01123 26.1502 3.017 26.1842 3.02454 26.2178C3.02649 26.2269 3.02917 26.2357 3.03143 26.2447C3.03851 26.2729 3.04663 26.3007 3.05609 26.3278C3.05951 26.3376 3.06305 26.3472 3.06677 26.3569C3.07727 26.3844 3.08897 26.4113 3.10187 26.4376C3.10571 26.4456 3.10925 26.4536 3.1134 26.4615C3.14838 26.5286 3.19087 26.5914 3.24005 26.6489C3.24762 26.6578 3.25562 26.6661 3.26349 26.6747C3.28107 26.694 3.29932 26.7125 3.31842 26.7303C3.32751 26.7388 3.33654 26.7472 3.34595 26.7554C3.36743 26.7739 3.38971 26.7914 3.41272 26.8081C3.41956 26.8131 3.42609 26.8186 3.4331 26.8234C3.46414 26.8448 3.49638 26.8645 3.52966 26.8823C3.53284 26.884 3.53619 26.8854 3.53937 26.887C3.56937 26.9026 3.60015 26.9166 3.63159 26.929C3.6405 26.9326 3.6496 26.9357 3.65869 26.939C3.68512 26.9486 3.71206 26.9571 3.7395 26.9645C3.75013 26.9674 3.76062 26.9702 3.7713 26.9727C3.79865 26.9791 3.82642 26.9841 3.85455 26.9882C3.86474 26.9897 3.87488 26.9917 3.88513 26.9929C3.92325 26.9975 3.96161 26.9998 4 27H26C26.2099 27 26.4144 26.9339 26.5847 26.8112C26.755 26.6885 26.8823 26.5153 26.9487 26.3162L30.5099 15.6324C30.6097 15.3318 30.6369 15.0117 30.5891 14.6985C30.5413 14.3853 30.42 14.0878 30.235 13.8306ZM11.6666 8L15.1333 10.6C15.4799 10.8588 15.9008 10.9991 16.3334 11H25V13H18.3027C17.9078 12.9996 17.5217 13.1165 17.1934 13.3359L14.6973 15H8.677C8.27699 14.9987 7.88587 15.118 7.55464 15.3422C7.2234 15.5665 6.96742 15.8853 6.82007 16.2572L5 20.8074V8H11.6666Z" fill="white"/>
                        </svg>
                    </Button>
                    <p className={styles.buttonText}>Tipos de archivos aceptados: .mp4</p>
                    <p className={styles.buttonText}>Tamaño máximo del archivo: 200 MB</p>
                    <p className={styles.buttonText}>Máximo número de archivos: 8</p>
                </div>
            )
        }

        {
            videos.length > 0 && (
                <div className={`${styles.videosContainer} ${videos.length > 3 && styles.videosContainerMargin}`}>
                    {
                        videos.map((video, index) => (
                            <div className={styles.videoContainer} key={index}>
                                <div className={styles.videoBackgroundContainer}>
                                    <video className={styles.video}>
                                        <source src={generateThumbnail(video)} type="video/mp4" />
                                    </video>
                                </div>
                                <svg
                                    className={styles.deleteIcon}
                                    width="36"
                                    height="36"
                                    viewBox="0 0 36 36"
                                    fill="none"
                                    xmlns="http://www.w3.org/2000/svg"
                                    onClick={() => handleDelete(index)}
                                >
                                    <path d="M12.375 4.5H23.625C23.9234 4.5 24.2095 4.38147 24.4205 4.1705C24.6315 3.95952 24.75 3.67337 24.75 3.375C24.75 3.07663 24.6315 2.79048 24.4205 2.5795C24.2095 2.36853 23.9234 2.25 23.625 2.25H12.375C12.0766 2.25 11.7905 2.36853 11.5795 2.5795C11.3685 2.79048 11.25 3.07663 11.25 3.375C11.25 3.67337 11.3685 3.95952 11.5795 4.1705C11.7905 4.38147 12.0766 4.5 12.375 4.5Z" fill="white"/>
                                    <path d="M30.375 6.75H5.625C5.32663 6.75 5.04048 6.86853 4.82951 7.07951C4.61853 7.29048 4.5 7.57663 4.5 7.875C4.5 8.17337 4.61853 8.45952 4.82951 8.67049C5.04048 8.88147 5.32663 9 5.625 9H6.75V29.25C6.75059 29.8466 6.98783 30.4185 7.40966 30.8403C7.83149 31.2622 8.40344 31.4994 9 31.5H27C27.5966 31.4994 28.1685 31.2622 28.5903 30.8403C29.0122 30.4185 29.2494 29.8466 29.25 29.25V9H30.375C30.6734 9 30.9595 8.88147 31.1705 8.67049C31.3815 8.45952 31.5 8.17337 31.5 7.875C31.5 7.57663 31.3815 7.29048 31.1705 7.07951C30.9595 6.86853 30.6734 6.75 30.375 6.75ZM15.75 23.625C15.75 23.9234 15.6315 24.2095 15.4205 24.4205C15.2095 24.6315 14.9234 24.75 14.625 24.75C14.3266 24.75 14.0405 24.6315 13.8295 24.4205C13.6185 24.2095 13.5 23.9234 13.5 23.625V14.625C13.5 14.3266 13.6185 14.0405 13.8295 13.8295C14.0405 13.6185 14.3266 13.5 14.625 13.5C14.9234 13.5 15.2095 13.6185 15.4205 13.8295C15.6315 14.0405 15.75 14.3266 15.75 14.625V23.625ZM22.5 23.625C22.5 23.9234 22.3815 24.2095 22.1705 24.4205C21.9595 24.6315 21.6734 24.75 21.375 24.75C21.0766 24.75 20.7905 24.6315 20.5795 24.4205C20.3685 24.2095 20.25 23.9234 20.25 23.625V14.625C20.25 14.3266 20.3685 14.0405 20.5795 13.8295C20.7905 13.6185 21.0766 13.5 21.375 13.5C21.6734 13.5 21.9595 13.6185 22.1705 13.8295C22.3815 14.0405 22.5 14.3266 22.5 14.625V23.625Z" fill="white"/>
                                </svg>
                                <p className={styles.videoName}>{video.name}</p>
                            </div>
                        ))
                    }
                    {
                        videos.length < 8 && (
                            <div className={styles.addVideo} onClick={handleUpload}>
                                <div className={styles.iconContainer}>
                                    <svg className={styles.addIcon} width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                                        <path d="M27 14.5H17.5V5C17.5 4.60218 17.342 4.22065 17.0607 3.93934C16.7794 3.65804 16.3978 3.5 16 3.5C15.6022 3.5 15.2206 3.65804 14.9393 3.93934C14.658 4.22064 14.5 4.60217 14.5 5V14.5H5C4.60217 14.5 4.22064 14.658 3.93934 14.9393C3.65804 15.2206 3.5 15.6022 3.5 16C3.5 16.3978 3.65804 16.7794 3.93934 17.0607C4.22065 17.342 4.60218 17.5 5 17.5H14.5V27C14.5 27.3978 14.658 27.7794 14.9393 28.0607C15.2206 28.342 15.6022 28.5 16 28.5C16.3978 28.5 16.7794 28.342 17.0607 28.0607C17.342 27.7794 17.5 27.3978 17.5 27V17.5H27C27.3978 17.5 27.7794 17.342 28.0607 17.0607C28.342 16.7794 28.5 16.3978 28.5 16C28.5 15.6022 28.342 15.2206 28.0607 14.9393C27.7794 14.658 27.3978 14.5 27 14.5Z" fill="white" stroke="white"/>
                                    </svg>
                                </div>
                                <p className={styles.addVideoText}>Agregar otro archivo</p>
                            </div>
                        )
                    }
                </div>
            )
        }

        <Button
            text={loading ? "Cargando..." : "Continuar"}
            color="black"
            disabled={videos.length == 0 || loading}
            className={styles.continueButton}
            onClick={() => UploadVideos()}
        />
    </div>
  )
}

export default UploadVideos
