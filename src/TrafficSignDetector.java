import org.opencv.core.*;
// N'importez PAS java.awt.Point si vous ne l'utilisez pas explicitement pour autre chose
// import java.awt.Point; // Supprimez ou commentez cette ligne si elle existe
import org.opencv.features2d.*;
import org.opencv.highgui.Highgui; // For imread
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;

import javax.swing.*;
import java.awt.*; // Nécessaire pour JFrame, JLabel, etc.
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import uk.co.caprica.vlcj.factory.MediaPlayerFactory;
import uk.co.caprica.vlcj.player.base.MediaPlayer;
import uk.co.caprica.vlcj.player.embedded.EmbeddedMediaPlayer;
import uk.co.caprica.vlcj.player.embedded.videosurface.callback.BufferFormat;
import uk.co.caprica.vlcj.player.embedded.videosurface.callback.BufferFormatCallback;
import uk.co.caprica.vlcj.player.embedded.videosurface.callback.RenderCallbackAdapter;
import uk.co.caprica.vlcj.player.component.EmbeddedMediaPlayerComponent;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.Timer;

public class TrafficSignDetector {

    // --- Configuration ---
    // HSV Red Color Range (adjust these)
    private static final Scalar LOWER_RED1 = new Scalar(0, 100, 100);
    private static final Scalar UPPER_RED1 = new Scalar(10, 255, 255);
    private static final Scalar LOWER_RED2 = new Scalar(160, 100, 100);
    private static final Scalar UPPER_RED2 = new Scalar(180, 255, 255);

    // Contour Filtering Parameters
    private static final double MIN_CONTOUR_AREA = 500;
    private static final double MAX_CONTOUR_AREA = 50000;
    private static final double MIN_ASPECT_RATIO = 0.7;
    private static final double MAX_ASPECT_RATIO = 1.3;

    // ORB Feature Detection and Matching
    private static final int ORB_NFEATURES = 500; // Avertissement: non utilisé avec l'API actuelle de 2.4.x
    private static final int BF_MATCHER_TYPE = DescriptorMatcher.BRUTEFORCE_HAMMING;
    private static final float GOOD_MATCH_PERCENT = 0.15f;
    private static final int MIN_MATCH_COUNT = 10;

    private static final String REFERENCE_SIGN_DIR = "reference_signs";
    private static final Size REFERENCE_SIGN_SIZE = new Size(64, 64);

    // --- Global OpenCV Objects ---
    private static FeatureDetector orbDetector;
    private static DescriptorExtractor orbDescriptorExtractor;
    private static DescriptorMatcher bfMatcher;
    private static Map<String, ReferenceSignData> referenceDb = new HashMap<>();

    // --- GUI Elements ---
    private static JFrame frame;
    private static JLabel imageLabel;
    private static JLabel infoLabel;
    private static String lastDisplayedSign = "None";

    // Ajout d'un BufferedImage pour la frame traitée
    private static BufferedImage processedImage;

    private static boolean referenceSignsLoaded = false;
    private static boolean videoLoaded = false;
    private static String videoErrorMessage = null;

    private static EmbeddedMediaPlayerComponent mediaPlayerComponent;
    private static JLabel processedLabel;

    // Helper class to store reference sign data
    static class ReferenceSignData {
        MatOfKeyPoint keypoints;
        Mat descriptors;
        Mat image; // For potential display or debugging
        String signValue;

        ReferenceSignData(String value, MatOfKeyPoint kp, Mat des, Mat img) {
            this.signValue = value;
            this.keypoints = kp;
            this.descriptors = des;
            this.image = img;
        }
    }

    static {
        // Load OpenCV native library
        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Native code library failed to load.\n" +
                               "Ensure a valid OpenCV native library for your platform " +
                               "(e.g., opencv_java2413.dll or .so) is in your java.library.path.\n" + e);
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
        if (mat.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = mat.channels() * mat.cols() * mat.rows();
        byte[] b = new byte[bufferSize];
        mat.get(0, 0, b); // get all the pixels
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(b, 0, targetPixels, 0, b.length);
        return image;
    }


    private static boolean loadReferenceSigns() {
        System.out.println("Loading reference signs...");
        File dir = new File(REFERENCE_SIGN_DIR);
        if (!dir.exists() || !dir.isDirectory()) {
            System.err.println("Error: Reference sign directory '" + REFERENCE_SIGN_DIR + "' not found.");
            return false;
        }

        orbDetector = FeatureDetector.create(FeatureDetector.ORB);
        orbDescriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.ORB);

        File[] files = dir.listFiles((_, name) -> name.toLowerCase().startsWith("ref_") && // CORRIGÉ ICI
                                                 (name.toLowerCase().endsWith(".png") || name.toLowerCase().endsWith(".jpg")));

        if (files == null || files.length == 0) {
            System.err.println("No reference signs found in " + REFERENCE_SIGN_DIR);
            return false;
        }

        for (File file : files) {
            try {
                String fileName = file.getName();
                String signValue = fileName.substring(fileName.indexOf('_') + 1, fileName.lastIndexOf('.'));
                Mat refImg = Highgui.imread(file.getAbsolutePath(), Highgui.CV_LOAD_IMAGE_COLOR);
                if (refImg.empty()) {
                    System.err.println("Warning: Could not load reference image " + file.getAbsolutePath());
                    continue;
                }

                Mat refImgResized = new Mat();
                Imgproc.resize(refImg, refImgResized, REFERENCE_SIGN_SIZE);
                Mat refGray = new Mat();
                Imgproc.cvtColor(refImgResized, refGray, Imgproc.COLOR_BGR2GRAY);

                MatOfKeyPoint keypoints = new MatOfKeyPoint();
                Mat descriptors = new Mat();

                orbDetector.detect(refGray, keypoints);
                orbDescriptorExtractor.compute(refGray, keypoints, descriptors);

                if (!descriptors.empty() && descriptors.rows() > 0) {
                    referenceDb.put(signValue, new ReferenceSignData(signValue, keypoints, descriptors, refImgResized));
                    System.out.println("Loaded reference sign: " + signValue + " with " + descriptors.rows() + " descriptors.");
                } else {
                    System.err.println("Warning: No descriptors found for reference sign " + signValue);
                }
            } catch (Exception e) {
                System.err.println("Error processing reference sign " + file.getName() + ": " + e.getMessage());
                e.printStackTrace();
            }
        }
        return !referenceDb.isEmpty();
    }

    private static Mat extractRedPixels(Mat imageBgr) {
        Mat hsvImage = new Mat();
        Imgproc.cvtColor(imageBgr, hsvImage, Imgproc.COLOR_BGR2HSV);

        Mat mask1 = new Mat();
        Mat mask2 = new Mat();
        Core.inRange(hsvImage, LOWER_RED1, UPPER_RED1, mask1);
        Core.inRange(hsvImage, LOWER_RED2, UPPER_RED2, mask2);

        Mat redMask = new Mat();
        Core.bitwise_or(mask1, mask2, redMask);

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.morphologyEx(redMask, redMask, Imgproc.MORPH_OPEN, kernel, new org.opencv.core.Point(-1,-1), 1); // CORRIGÉ ICI
        Imgproc.morphologyEx(redMask, redMask, Imgproc.MORPH_CLOSE, kernel, new org.opencv.core.Point(-1,-1), 1); // CORRIGÉ ICI
        
        hsvImage.release();
        mask1.release();
        mask2.release();
        kernel.release();

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

    private static RecognizedSign recognizeSign(Mat roiBgr) {
        if (roiBgr == null || roiBgr.empty()) {
            return null;
        }

        Mat roiResized = new Mat();
        Imgproc.resize(roiBgr, roiResized, REFERENCE_SIGN_SIZE);
        Mat roiGray = new Mat();
        Imgproc.cvtColor(roiResized, roiGray, Imgproc.COLOR_BGR2GRAY);

        MatOfKeyPoint kpRoi = new MatOfKeyPoint();
        Mat desRoi = new Mat();

        orbDetector.detect(roiGray, kpRoi);
        orbDescriptorExtractor.compute(roiGray, kpRoi, desRoi);

        if (desRoi.empty() || kpRoi.toArray().length < (MIN_MATCH_COUNT / (GOOD_MATCH_PERCENT + 1e-6))) {
            roiResized.release();
            roiGray.release();
            kpRoi.release();
            desRoi.release();
            return null;
        }

        String bestMatchSignValue = null;
        int maxGoodMatches = 0;

        for (Map.Entry<String, ReferenceSignData> entry : referenceDb.entrySet()) {
            ReferenceSignData refData = entry.getValue();
            MatOfDMatch matches = new MatOfDMatch();
            
            if (refData.descriptors.empty() || desRoi.empty() || 
                refData.descriptors.type() != CvType.CV_8U || desRoi.type() != CvType.CV_8U) {
                continue; 
            }

            bfMatcher.match(refData.descriptors, desRoi, matches);

            List<DMatch> matchesList = matches.toList();
            if (matchesList.isEmpty()) {
                matches.release();
                continue;
            }
            
            Collections.sort(matchesList, (m1, m2) -> Float.compare(m1.distance, m2.distance));

            int numGoodMatches = (int) (matchesList.size() * GOOD_MATCH_PERCENT);
            if (numGoodMatches == 0 && matchesList.size() > 0) numGoodMatches = 1; 

            List<DMatch> goodMatchesSubList = matchesList.subList(0, Math.min(numGoodMatches, matchesList.size()));

            if (goodMatchesSubList.size() > maxGoodMatches && goodMatchesSubList.size() >= MIN_MATCH_COUNT) {
                maxGoodMatches = goodMatchesSubList.size();
                bestMatchSignValue = refData.signValue;
            }
            matches.release();
        }
        
        roiResized.release();
        roiGray.release();
        kpRoi.release();
        desRoi.release();

        if (bestMatchSignValue != null) {
            return new RecognizedSign(bestMatchSignValue, maxGoodMatches);
        }
        return null;
    }
    
    static class RecognizedSign {
        String value;
        int matchCount;
        RecognizedSign(String v, int mc) {
            this.value = v;
            this.matchCount = mc;
        }
    }

    private static ProcessedFrameData processFrame(Mat frame) {
        if (frame.empty()) {
            return new ProcessedFrameData(null, "No Frame");
        }

        Mat displayFrame = frame.clone(); 

        Mat redMask = extractRedPixels(frame);
        List<MatOfPoint> potentialContours = findPotentialSignContours(redMask);
        redMask.release(); 

        String detectedSignText = "None";

        for (MatOfPoint contour : potentialContours) {
            Rect roiRect = Imgproc.boundingRect(contour);
            Core.rectangle(displayFrame, roiRect.tl(), roiRect.br(), new Scalar(0, 255, 0), 2);
            Mat roi = new Mat(frame, roiRect); 

            RecognizedSign recognized = recognizeSign(roi);
            roi.release(); 

            if (recognized != null) {
                detectedSignText = recognized.value; 
                Core.putText(displayFrame, recognized.value,
                                new org.opencv.core.Point(roiRect.x, roiRect.y - 10),
                                Core.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(0, 0, 255), 2);
                break; 
            }
        }
        
        for(MatOfPoint mop : potentialContours) {
            mop.release();
        }

        return new ProcessedFrameData(displayFrame, detectedSignText);
    }

    static class ProcessedFrameData {
        Mat frame;
        String detectedSign;
        ProcessedFrameData(Mat f, String ds) {
            this.frame = f;
            this.detectedSign = ds;
        }
    }

    private static void initGui() {
        frame = new JFrame("Traffic Sign Detection - Java OpenCV + VLCJ");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new GridLayout(1, 2)); // 2 colonnes : vidéo originale + traitée

        mediaPlayerComponent = new EmbeddedMediaPlayerComponent();
        frame.add(mediaPlayerComponent);

        processedLabel = new JLabel();
        processedLabel.setHorizontalAlignment(SwingConstants.CENTER);
        frame.add(processedLabel);

        infoLabel = new JLabel("Detected: None | FPS: 0");
        infoLabel.setHorizontalAlignment(SwingConstants.CENTER);
        frame.add(infoLabel, BorderLayout.SOUTH);

        frame.setSize(1400, 700);
        frame.setVisible(true);
    }
    
    private static void updateGui(Mat displayMat, String detectedSign, double fps) {
        if (displayMat != null && !displayMat.empty()) {
            BufferedImage bufImage = matToBufferedImage(displayMat);
            imageLabel.setIcon(new ImageIcon(bufImage));
        }
        infoLabel.setText(String.format("Detected: %s | FPS: %.1f", detectedSign, fps));
        frame.repaint(); 
    }

    private static void showWarning(String message) {
        if (infoLabel != null) {
            infoLabel.setText("<html><span style='color:orange;'>" + message + "</span></html>");
        }
    }

    public static void main(String[] args) {
        referenceSignsLoaded = loadReferenceSigns();
        bfMatcher = DescriptorMatcher.create(BF_MATCHER_TYPE);
        if (bfMatcher == null) {
            System.err.println("Failed to create BFMatcher. Check OpenCV setup.");
            return;
        }
        initGui();

        if (!referenceSignsLoaded) {
            showWarning("Aucun panneau de référence valide trouvé dans le dossier 'reference_signs'.<br>Ajoutez des images nommées ref_XX.jpg ou ref_XX.png.");
        }

        // --- VLCJ 4.x INTEGRATION + DÉTECTION EN TEMPS RÉEL ---
        String videoPath = "video1.avi";
        java.io.File videoFile = new java.io.File(videoPath);
        if (!videoFile.exists()) {
            showWarning("Fichier vidéo '" + videoPath + "' introuvable. Placez une vidéo compatible dans le dossier du projet.");
            return;
        }
        mediaPlayerComponent.mediaPlayer().media().play(videoPath);

        // Timer pour la détection en temps réel (toutes les 100 ms)
        Timer detectionTimer = new Timer(100, new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if (mediaPlayerComponent.mediaPlayer().status().isPlaying()) {
                    BufferedImage snap = mediaPlayerComponent.mediaPlayer().snapshots().get();
                    if (snap != null) {
                        Mat mat = bufferedImageToMat(snap);
                        ProcessedFrameData pfd = processFrame(mat);
                        if (pfd != null && pfd.frame != null && !pfd.frame.empty()) {
                            BufferedImage bufImage = matToBufferedImage(pfd.frame);
                            processedLabel.setIcon(new ImageIcon(bufImage));
                            infoLabel.setText(String.format("Detected: %s", pfd.detectedSign));
                            pfd.frame.release();
                        }
                        mat.release();
                    }
                }
            }
        });
        detectionTimer.start();
    }
}
