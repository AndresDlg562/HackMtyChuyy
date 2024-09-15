import styles from "./button.module.css";

const UploadVideos = ({text, color, onClick, className, disabled, children}: {text: string, color: "black" | "white", onClick?: () => void, className?: any, disabled?: boolean, children?: React.ReactNode}) => {
    return (
        <button
            className={`${styles.button} ${styles[color]} ${disabled && styles.disabled} ${className}`}
            disabled={disabled}
            onClick={onClick}
        >
            {
                children && children
            }
            {text}
        </button>
    )
}

export default UploadVideos
