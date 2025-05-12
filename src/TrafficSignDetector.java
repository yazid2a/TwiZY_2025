import org.opencv.core.*;
//N'importez PAS java.awt.Point si vous ne l'utilisez pas explicitement pour autre chose
//import java.awt.Point; // Supprimez ou commentez cette ligne si elle existe
import org.opencv.features2d.*;
import org.opencv.highgui.Highgui; // For imread
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*; // Nécessaire pour JFrame, JLabel, etc.
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FilenameFilter;
import java.io.InputStream;
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

class Circle{
	public org.opencv.core.Point center;
	public int radius;

	public Circle(org.opencv.core.Point c, int r) {
		this.center = c;
		this.radius = r;
	}
}

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
 private static Map<String, Mat> referenceDb = new HashMap<>();

 // --- GUI Elements ---
 private static JFrame frame;
 private static JLabel imageLabel;
 private static JLabel infoLabel;
 private static String DisplayText = "None";

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
/*
 class Circle{
		public org.opencv.core.Point center;
		public int radius;

		public Circle(org.opencv.core.Point c, int r) {
			this.center = c;
			this.radius = r;
		}
	}
 */
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

	     FilenameFilter filter = new FilenameFilter() {
	    	 public boolean accept(File dir, String name) {
	    		 return name.toLowerCase().startsWith("ref_") && // CORRIGÉ ICI
	                     (name.toLowerCase().endsWith(".png") || name.toLowerCase().endsWith(".jpg"));
	    	 }
	     };
	     
	     File[] files = dir.listFiles(filter);

	     if (files == null || files.length == 0) {
	         System.err.println("No reference signs found in " + REFERENCE_SIGN_DIR);
	         return false;
	     }

	     for (File file : files) {
            String fileName = file.getName();
            String signValue = fileName.substring(fileName.indexOf('_') + 1, fileName.lastIndexOf('.'));
            Mat refImg = Highgui.imread(file.getAbsolutePath(), Highgui.CV_LOAD_IMAGE_COLOR);
            if (refImg.empty()) {
                System.err.println("Warning: Could not load reference image " + file.getAbsolutePath());
                continue;
            }
            referenceDb.put(signValue, rgb2gray(refImg));
	     }
	     return true;
	}
 
 static class RecognizedSign {
     String value;
     int matchCount;
     RecognizedSign(String v, int mc) {
         this.value = v;
         this.matchCount = mc;
     }
 }

	
	
	//--------------------------------------------------------------
	
	public static Mat rgb2hsv(Mat src) {
		Mat dst = new Mat();
		Imgproc.cvtColor(src, dst, Imgproc.COLOR_RGB2HSV);
		return dst;
	}
	
	public static Mat rgb2gray(Mat src) {
		Mat dst = new Mat();
		Imgproc.cvtColor(src, dst, Imgproc.COLOR_RGB2GRAY);
		return dst;
	}
	
	public static ArrayList<MatOfPoint> detectContours(Mat imgGray){
		Imgproc.GaussianBlur(imgGray, imgGray, new Size (11, 11), 3);
		int tresh = 100;
		Mat canny_output = new Mat();
		ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		Mat hierarchy = new Mat();
		Imgproc.Canny(imgGray, canny_output, tresh, tresh*2);
		Imgproc.findContours(canny_output, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
		
		// affichage des contours
		/*
		Mat drawing = Mat.zeros(canny_output.size(), CvType.CV_8UC3);
		for (int i = 0; i<contours.size(); i++) {
			Imgproc.drawContours(drawing, contours, i, new Scalar(255, 255, 255));
		}
		ImShow("e", drawing);
		*/
		return contours;
	}
	
	public static ArrayList<Circle> findCircles(Mat img, ArrayList<MatOfPoint> contours){
		ArrayList<Circle> circles = new ArrayList<Circle>();
		MatOfPoint2f matOfPoint2f = new MatOfPoint2f();
		for (int c=0; c<contours.size(); c++) {
			float[] radius = new float[1];
			org.opencv.core.Point center = new org.opencv.core.Point(); // pas touche sinon il casse les pieds
			MatOfPoint contour = contours.get(c);
			double contourArea = Imgproc.contourArea(contour);
			matOfPoint2f.fromList(contour.toList());
			Imgproc.minEnclosingCircle(matOfPoint2f, center, radius);
			if ((contourArea/(Math.PI*radius[0]*radius[0]))>=0.8 && radius[0]>15) {
				circles.add(new Circle(center, (int)radius[0]));
			}
		}
		return circles;
	}

	//l'identification se fait en niveau de gris         
	public static String identifiePanneau(Mat panneau, Map<String, Mat> panneaux_ref) {
		int best_choice = Integer.MAX_VALUE;
		String best_panneau = null;
		Imgproc.GaussianBlur(panneau, panneau, new Size (11, 11), 2);
		for (String panneau_ref_nom : panneaux_ref.keySet()) {
			int produit_conv = 0;
			Mat panneau_ref_img = panneaux_ref.get(panneau_ref_nom);
			Imgproc.GaussianBlur(panneau_ref_img, panneau_ref_img, new Size (11, 11), 2);
			Size taille = panneau.size();
			Mat panneau_ref_img_cropped = new Mat();
			Imgproc.resize(panneau_ref_img, panneau_ref_img_cropped, taille);

			int result = 0;
			int L = (int) panneau.size().width;
			int H = (int) panneau_ref_img_cropped.size().height;
			for (int l = 0; l<L; l++){
				for (int h = 0; h<H; h++){
					double[] data = panneau.get(l, h);
					double[] data_ref = panneau_ref_img_cropped.get(l, h);
					result = (int) (result + Math.pow(Math.abs(data_ref[0] - data[0]), 3)*0.0001);

				}
				
				produit_conv = produit_conv + result;
			}
			produit_conv = produit_conv/3;
			if (produit_conv<best_choice){
				best_choice = produit_conv;
				best_panneau = panneau_ref_nom;
			}
		}
		return best_panneau;
		
	}
 
 private static ProcessedFrameData processFrame(Mat frame) {
     if (frame.empty()) {
         return new ProcessedFrameData(null, "No Frame");
     }

    Mat displayFrame = frame.clone();
     
    Mat imgHSV = rgb2hsv(displayFrame);
	Mat threshold_img = new Mat();
	Core.inRange(imgHSV, new Scalar(100, 80, 0), new Scalar(150, 255, 255), threshold_img);
	ArrayList<MatOfPoint> contours = detectContours(threshold_img);
	ArrayList<Circle> panneaux_trouves = findCircles(displayFrame, contours);
	String detectedSignText = "None";
	for (Circle c : panneaux_trouves) {
		Core.circle(displayFrame, c.center, c.radius, new Scalar(0, 255, 0), 2);
		
		int x = (int) c.center.x;
		int y = (int) c.center.y;
		int dx = c.radius;
		int dy = c.radius;
		Mat tmp = frame.submat(y-dy, y+dy, x-dx, x+dx);
		Mat imgPanneau = Mat.zeros(tmp.size(), tmp.type());
		tmp.copyTo(imgPanneau);
		detectedSignText = identifiePanneau(rgb2gray(imgPanneau), referenceDb);
	}
     return new ProcessedFrameData(displayFrame, detectedSignText);
 }

 //--------------------------------------------------
 
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
                         if (!pfd.detectedSign.equals("None")) {
                        	 DisplayText = pfd.detectedSign;
                        	 infoLabel.setText(String.format("Detected: %s", DisplayText));
                         }
                         else {
                        	 infoLabel.setText(String.format("Detected: %s", DisplayText.concat(" (Last seen)")));
                         }
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
