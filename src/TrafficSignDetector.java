import org.opencv.core.*;
import org.opencv.core.Point; // Explicitement importé pour clarté
// import org.opencv.features2d.ORB; // Commenté car ORB n'est pas utilisé par défaut
// import org.opencv.features2d.DescriptorMatcher; // Commenté
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.util.ArrayList;
// import java.util.Collections; // Non utilisé si ORB est désactivé
// import java.util.HashMap; // Non utilisé si ORB est désactivé
import java.util.List;
// import java.util.Map; // Non utilisé si ORB est désactivé
import uk.co.caprica.vlcj.player.component.EmbeddedMediaPlayerComponent;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.Timer;
import org.json.JSONObject;
import java.io.ByteArrayOutputStream;
import java.net.HttpURLConnection;
import java.io.DataOutputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.URL;
import javax.imageio.ImageIO;


public class TrafficSignDetector {

    // --- Configuration ---
    private static final Scalar LOWER_RED1 = new Scalar(0, 100, 100);
    private static final Scalar UPPER_RED1 = new Scalar(10, 255, 255);
    private static final Scalar LOWER_RED2 = new Scalar(160, 100, 100);
    private static final Scalar UPPER_RED2 = new Scalar(180, 255, 255);

    private static final double MIN_CONTOUR_AREA = 500;
    private static final double MAX_CONTOUR_AREA = 50000;
    private static final double MIN_ASPECT_RATIO = 0.7;
    private static final double MAX_ASPECT_RATIO = 1.3;

    // --- Configuration CNN ---
    private static final String CNN_MODEL_PATH_ONNX = "gtsrb_model.onnx"; // CHARGE CE FICHIER
    private static final int CNN_INPUT_WIDTH = 48;
    private static final int CNN_INPUT_HEIGHT = 48;
    private static final double CNN_SCALE_FACTOR = 1.0 / 255.0;
    private static final Scalar CNN_MEAN_SUBTRACTION = new Scalar(0, 0, 0);
    private static final boolean CNN_SWAP_RB = false;
    private static final boolean CNN_CROP = false;
    private static final double CNN_CONFIDENCE_THRESHOLD = 0.2; // Gardé bas pour débogage

    // --- Global OpenCV Objects ---
    private static Net signRecognitionNet;
    private static List<String> classNames;
    private static boolean cnnModelLoaded = false;

    // --- GUI Elements ---
    private static JFrame frame;
    private static JLabel infoLabel;
    private static JLabel signValueLabel;
    private static String lastDisplayedSign = "None";
    private static EmbeddedMediaPlayerComponent mediaPlayerComponent;
    private static JLabel processedLabel;

    static {
        try {
            System.loadLibrary("opencv_java4110"); // VÉRIFIEZ CE NOM (ex: opencv_java4110)
            System.out.println("OpenCV Native Library 'opencv_java4110' loaded successfully.");
            System.out.println("OpenCV Version: " + Core.VERSION);
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Native code library failed to load.\n" +
                               "Ensure opencv_java4110.dll is in java.library.path.\n" + e);
            System.err.println("Current java.library.path: " + System.getProperty("java.library.path"));
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static Mat bufferedImageToMat(BufferedImage bi) {
        Mat mat = new Mat(bi.getHeight(), bi.getWidth(), CvType.CV_8UC3);
        byte[] data = ((DataBufferByte) bi.getRaster().getDataBuffer()).getData();
        mat.put(0, 0, data);
        return mat;
    }

    public static BufferedImage matToBufferedImage(Mat mat) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (mat.channels() > 1) { type = BufferedImage.TYPE_3BYTE_BGR; }
        int bufferSize = mat.channels() * mat.cols() * mat.rows();
        byte[] b = new byte[bufferSize];
        mat.get(0, 0, b);
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(b, 0, targetPixels, 0, b.length);
        return image;
    }

    private static Mat extractRedPixels(Mat imageBgr) {
        Mat hsvImage = new Mat();
        Imgproc.cvtColor(imageBgr, hsvImage, Imgproc.COLOR_BGR2HSV);
        Mat mask1 = new Mat(); Mat mask2 = new Mat();
        Core.inRange(hsvImage, LOWER_RED1, UPPER_RED1, mask1);
        Core.inRange(hsvImage, LOWER_RED2, UPPER_RED2, mask2);
        Mat redMask = new Mat();
        Core.bitwise_or(mask1, mask2, redMask);
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.morphologyEx(redMask, redMask, Imgproc.MORPH_OPEN, kernel, new Point(-1, -1), 1);
        Imgproc.morphologyEx(redMask, redMask, Imgproc.MORPH_CLOSE, kernel, new Point(-1, -1), 1);
        hsvImage.release(); mask1.release(); mask2.release(); kernel.release();
        return redMask;
    }

    private static List<MatOfPoint> findPotentialSignContours(Mat redMask) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(redMask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        hierarchy.release();
        List<MatOfPoint> potentialSigns = new ArrayList<>();
        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area > MIN_CONTOUR_AREA && area < MAX_CONTOUR_AREA) {
                Rect boundingRect = Imgproc.boundingRect(contour);
                double aspectRatio = (double) boundingRect.width / boundingRect.height;
                if (aspectRatio > MIN_ASPECT_RATIO && aspectRatio < MAX_ASPECT_RATIO) {
                    potentialSigns.add(contour);
                }
            }
        }
        return potentialSigns;
    }

    private static void initializeClassNames() {
        classNames = new ArrayList<>();
        classNames.add("Speed limit (20km/h)"); classNames.add("Speed limit (30km/h)");
        classNames.add("Speed limit (50km/h)"); classNames.add("Speed limit (60km/h)");
        classNames.add("Speed limit (70km/h)"); classNames.add("Speed limit (80km/h)");
        classNames.add("End of speed limit (80km/h)"); classNames.add("Speed limit (100km/h)");
        classNames.add("Speed limit (120km/h)"); classNames.add("No passing");
        classNames.add("No passing for vehicles over 3.5 metric tons");
        classNames.add("Right-of-way at the next intersection"); classNames.add("Priority road");
        classNames.add("Yield"); classNames.add("Stop"); classNames.add("No vehicles");
        classNames.add("Vehicles over 3.5 metric tons prohibited"); classNames.add("No entry");
        classNames.add("General caution"); classNames.add("Dangerous curve to the left");
        classNames.add("Dangerous curve to the right"); classNames.add("Double curve");
        classNames.add("Bumpy road"); classNames.add("Slippery road");
        classNames.add("Road narrows on the right"); classNames.add("Road work");
        classNames.add("Traffic signals"); classNames.add("Pedestrians");
        classNames.add("Children crossing"); classNames.add("Bicycles crossing");
        classNames.add("Beware of ice/snow"); classNames.add("Wild animals crossing");
        classNames.add("End of all speed and passing limits"); classNames.add("Turn right ahead");
        classNames.add("Turn left ahead"); classNames.add("Ahead only");
        classNames.add("Go straight or right"); classNames.add("Go straight or left");
        classNames.add("Keep right"); classNames.add("Keep left");
        classNames.add("Roundabout mandatory"); classNames.add("End of no passing");
        classNames.add("End of no passing by vehicles over 3.5 metric tons");
        if (classNames.size() != 43) { System.err.println("ERREUR: classNames doit avoir 43 éléments!"); }
    }

    private static boolean loadSignRecognitionModelCNN() {
        System.out.println("Loading CNN model for sign recognition from: " + CNN_MODEL_PATH_ONNX);
        File modelFile = new File(CNN_MODEL_PATH_ONNX);
        if (!modelFile.exists() || !modelFile.isFile()) {
            System.err.println("Error: CNN model file not found: " + modelFile.getAbsolutePath());
            return false;
        }
        try {
            signRecognitionNet = Dnn.readNetFromONNX(CNN_MODEL_PATH_ONNX); // CHARGE LE MODÈLE ONNX
            if (signRecognitionNet.empty()) {
                System.err.println("Error: Could not load CNN model (ONNX). Network is empty.");
                return false;
            }
            System.out.println("CNN model (ONNX) loaded successfully.");
            initializeClassNames();
            return true;
        } catch (Exception e) {
            System.err.println("Exception while loading CNN model (ONNX): " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    private static RecognizedSign recognizeSignWithCNN(Mat roiBgr) {
        if (roiBgr == null || roiBgr.empty() || signRecognitionNet == null || signRecognitionNet.empty() || classNames == null || classNames.isEmpty()) {
            return null;
        }
        Mat preprocessedRoi = new Mat();
        Imgproc.resize(roiBgr, preprocessedRoi, new Size(CNN_INPUT_WIDTH, CNN_INPUT_HEIGHT));
        Mat blob = Dnn.blobFromImage(preprocessedRoi, CNN_SCALE_FACTOR,
                                     new Size(CNN_INPUT_WIDTH, CNN_INPUT_HEIGHT),
                                     CNN_MEAN_SUBTRACTION, CNN_SWAP_RB, CNN_CROP, CvType.CV_32F);
        if (blob.empty()) {
            System.err.println("recognizeSignWithCNN: Blob is empty.");
            preprocessedRoi.release(); return null;
        }
        try {
            signRecognitionNet.setInput(blob);
            Mat output = signRecognitionNet.forward();
            if (output.rows() == 1 && output.cols() == classNames.size()) {
                Core.MinMaxLocResult mm = Core.minMaxLoc(output);
                double confidence = mm.maxVal;
                int predictedClassId = (int) mm.maxLoc.x;
                System.out.println("CNN Raw Prediction - Class ID: " + predictedClassId +
                                   ", Name: " + (predictedClassId >= 0 && predictedClassId < classNames.size() ? classNames.get(predictedClassId) : "ID_OOB") +
                                   ", Confidence: " + String.format("%.4f", confidence));
                output.release();
                if (confidence > CNN_CONFIDENCE_THRESHOLD) {
                    if (predictedClassId >= 0 && predictedClassId < classNames.size()) {
                        System.out.println(">>> CNN Recognized: " + classNames.get(predictedClassId) + " (" + String.format("%.2f", confidence*100) + "%)");
                        return new RecognizedSign(classNames.get(predictedClassId), (int) (confidence * 100));
                    } else {
                        System.err.println("CNN Predicted class ID out of bounds: " + predictedClassId);
                    }
                }
            } else {
                System.err.println("CNN output Mat shape unexpected: " + output.size().toString() +
                                   " Expected: 1x" + classNames.size());
                output.release();
            }
        } catch (Exception e) {
            System.err.println("Exception during CNN forward pass: " + e.getMessage());
        } finally {
            blob.release();
            preprocessedRoi.release();
        }
        return null;
    }

    static class RecognizedSign {
        String value; int matchCount;
        RecognizedSign(String v, int mc) { this.value = v; this.matchCount = mc; }
    }

    private static ProcessedFrameData processFrame(Mat frame) {
        if (frame.empty()) { return new ProcessedFrameData(null, "No Frame"); }
        Mat displayFrame = frame.clone();
        Mat redMask = extractRedPixels(frame);
        List<MatOfPoint> potentialContours = findPotentialSignContours(redMask);
        redMask.release();
        String bestDetectedSignTextThisFrame = "None";
        int bestConfidenceThisFrame = 0;
        Rect bestRectThisFrame = null;

        for (MatOfPoint contour : potentialContours) {
            Rect roiRect = Imgproc.boundingRect(contour);
            Imgproc.rectangle(displayFrame, roiRect.tl(), roiRect.br(), new Scalar(0, 255, 0), 2);
            Mat roi = new Mat(frame, roiRect);
            RecognizedSign currentRecognizedSign = null;
            if (cnnModelLoaded) {
                currentRecognizedSign = recognizeSignWithCNN(roi);
            }
            roi.release();
            if (currentRecognizedSign != null) {
                if (currentRecognizedSign.matchCount > bestConfidenceThisFrame) {
                    bestConfidenceThisFrame = currentRecognizedSign.matchCount;
                    bestDetectedSignTextThisFrame = currentRecognizedSign.value;
                    bestRectThisFrame = roiRect;
                }
            }
        }
        if (bestRectThisFrame != null && !bestDetectedSignTextThisFrame.equals("None")) {
            String textToDisplay = bestDetectedSignTextThisFrame + " (" + bestConfidenceThisFrame + "%)";
            Imgproc.putText(displayFrame, textToDisplay,
                            new Point(bestRectThisFrame.x, bestRectThisFrame.y - 10),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 0, 255), 2);
        }
        for (MatOfPoint mop : potentialContours) { mop.release(); }
        return new ProcessedFrameData(displayFrame, bestDetectedSignTextThisFrame);
    }

    static class ProcessedFrameData {
        Mat frame; String detectedSign;
        ProcessedFrameData(Mat f, String ds) { this.frame = f; this.detectedSign = ds; }
    }

    private static void initGui() {
        frame = new JFrame("Traffic Sign Detection - Java OpenCV + VLCJ");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        JPanel mainPanel = new JPanel(new BorderLayout());
        frame.add(mainPanel);
        JPanel videoPanel = new JPanel(new GridLayout(1, 2));
        mediaPlayerComponent = new EmbeddedMediaPlayerComponent();
        videoPanel.add(mediaPlayerComponent);
        processedLabel = new JLabel();
        processedLabel.setHorizontalAlignment(SwingConstants.CENTER);
        videoPanel.add(processedLabel);
        mainPanel.add(videoPanel, BorderLayout.CENTER);
        JPanel infoPanel = new JPanel(new GridLayout(2, 1));
        signValueLabel = new JLabel("Panneau détecté : Aucun");
        signValueLabel.setFont(new Font("Arial", Font.BOLD, 16));
        signValueLabel.setHorizontalAlignment(SwingConstants.CENTER);
        infoPanel.add(signValueLabel);
        infoLabel = new JLabel("FPS: 0");
        infoLabel.setHorizontalAlignment(SwingConstants.CENTER);
        infoPanel.add(infoLabel);
        mainPanel.add(infoPanel, BorderLayout.SOUTH);
        frame.setSize(1400, 700);
        frame.setVisible(true);
    }

    private static void showWarning(String message) {
        if (infoLabel != null) {
            infoLabel.setText("<html><span style='color:orange;'>" + message + "</span></html>");
        } else if (signValueLabel != null) {
             signValueLabel.setText("<html><span style='color:orange;'>" + message + "</span></html>");
        }
    }

    // Appel API Python pour la détection
    public static String detectSignWithApi(BufferedImage image) {
        try {
            // Convertir l'image en PNG dans un ByteArrayOutputStream
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ImageIO.write(image, "png", baos);
            byte[] imageBytes = baos.toByteArray();

            // Préparer la requête HTTP POST multipart
            String boundary = "----WebKitFormBoundary" + System.currentTimeMillis();
            URL url = new URL("http://127.0.0.1:8000/predict");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setDoOutput(true);
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);

            DataOutputStream request = new DataOutputStream(conn.getOutputStream());
            request.writeBytes("--" + boundary + "\r\n");
            request.writeBytes("Content-Disposition: form-data; name=\"file\"; filename=\"frame.png\"\r\n");
            request.writeBytes("Content-Type: image/png\r\n\r\n");
            request.write(imageBytes);
            request.writeBytes("\r\n--" + boundary + "--\r\n");
            request.flush();
            request.close();

            // Lire la réponse
            int responseCode = conn.getResponseCode();
            if (responseCode == HttpURLConnection.HTTP_OK) {
                BufferedReader in = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                String inputLine;
                StringBuilder response = new StringBuilder();
                while ((inputLine = in.readLine()) != null) {
                    response.append(inputLine);
                }
                in.close();

                // Parser le JSON
                JSONObject json = new JSONObject(response.toString());
                String className = json.getString("class_name");
                double confidence = json.getDouble("confidence");
                return className + String.format(" (%.1f%%)", confidence * 100);
            } else {
                return "Erreur API: " + responseCode;
            }
        } catch (Exception e) {
            e.printStackTrace();
            return "Erreur API";
        }
    }

    public static void main(String[] args) {
        cnnModelLoaded = loadSignRecognitionModelCNN();
        initGui();
        if (!cnnModelLoaded) {
            showWarning("Modèle CNN non chargé. L'identification sera désactivée.");
        }
        String videoPath = "video2.avi";
        File videoFile = new File(videoPath);
        if (!videoFile.exists()) {
            showWarning("Fichier vidéo '" + videoPath + "' introuvable: " + videoFile.getAbsolutePath());
        } else {
            mediaPlayerComponent.mediaPlayer().media().play(videoFile.getAbsolutePath());
        }
        Timer detectionTimer = new Timer(100, new ActionListener() {
            long lastFrameTimeNano = System.nanoTime();
            String lastApiResult = "Aucun panneau détecté";
            @Override
            public void actionPerformed(ActionEvent e) {
                if (mediaPlayerComponent.mediaPlayer().status().isPlaying()) {
                    BufferedImage snap = mediaPlayerComponent.mediaPlayer().snapshots().get();
                    if (snap != null) {
                        Mat mat = bufferedImageToMat(snap);
                        Mat redMask = extractRedPixels(mat);
                        List<MatOfPoint> potentialContours = findPotentialSignContours(redMask);
                        redMask.release();

                        String apiResult = null;
                        BufferedImage roiImage = null;

                        double maxArea = 0;
                        Rect bestRect = null;
                        for (MatOfPoint contour : potentialContours) {
                            double area = Imgproc.contourArea(contour);
                            if (area > maxArea) {
                                maxArea = area;
                                bestRect = Imgproc.boundingRect(contour);
                            }
                        }

                        if (bestRect != null) {
                            Imgproc.rectangle(mat, bestRect.tl(), bestRect.br(), new Scalar(0, 255, 0), 2);
                            Mat roiMat = new Mat(mat, bestRect);
                            roiImage = matToBufferedImage(roiMat);
                            apiResult = detectSignWithApi(roiImage);
                            roiMat.release();
                        }

                        BufferedImage bufImage = matToBufferedImage(mat);
                        processedLabel.setIcon(new ImageIcon(bufImage));
                        if (apiResult != null && !apiResult.startsWith("Erreur") && !apiResult.equals("Aucun panneau détecté")) {
                            signValueLabel.setText("<html><span style='color:green;'>Panneau détecté : " + apiResult + "</span></html>");
                            lastApiResult = apiResult;
                        } else if (apiResult == null) {
                            // Ne rien changer, garder le dernier panneau détecté
                        } else if (apiResult.startsWith("Erreur")) {
                            signValueLabel.setText("<html><span style='color:red;'>" + apiResult + "</span></html>");
                        }
                        long currentTimeNano = System.nanoTime();
                        double fps = 1_000_000_000.0 / (currentTimeNano - lastFrameTimeNano);
                        lastFrameTimeNano = currentTimeNano;
                        infoLabel.setText(String.format("FPS: %.1f", fps));

                        mat.release();
                        for (MatOfPoint mop : potentialContours) mop.release();
                    }
                }
            }
        });
        if (cnnModelLoaded || videoFile.exists()) {
            detectionTimer.start();
        } else {
            showWarning("Ni modèle CNN ni vidéo chargés. Timer non démarré.");
        }
    }
}
