package com.example.kevin.opencv_takephoto;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.ListIterator;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.hardware.Camera.Size;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SubMenu;
import android.view.SurfaceView;
import android.view.View;
import android.view.View.OnTouchListener;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.Toast;

public class MainActivity extends Activity implements CvCameraViewListener2, OnTouchListener {
    static {
        System.loadLibrary("native-lib");
    }
    private static final String TAG = "OCVSample::Activity";

    private Tutorial3View mOpenCvCameraView;
    private List<Size> mResolutionList;
    private MenuItem[] mEffectMenuItems;
    private SubMenu mColorEffectsMenu;
    private MenuItem[] mResolutionMenuItems;
    private SubMenu mResolutionMenu;
    private boolean need_light_enhance;
    private Button light_enhance_bt;
    private Mat rgb;
    private Mat ycrcb;
    private ArrayList<Mat> channels;
    private Mat cr;
    private Mat mask;
    private Mat res;
    private int count;
    private Mat detect;
    private ArrayList<MatOfPoint> contours;
    private final float fx = 1.0f / 16;
    private final float arg_a = -0.3293f;
    private final float arg_b = 1.1258f;
    private native void matrixPow(long m1Data, long m2Data,long dstData, int size);
    private native void copyRGB(long rgbData,long rgbaData,int rows,int cols);
    private native void getK(long tData,long outData, int h, int w);
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.setOnTouchListener(MainActivity.this);
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public MainActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());

    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (Tutorial3View) findViewById(R.id.tutorial3_activity_java_surface_view);

        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

        mOpenCvCameraView.setCvCameraViewListener(this);

        need_light_enhance = false;
        light_enhance_bt = (Button)findViewById(R.id.light_enhance_bt);
        light_enhance_bt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                need_light_enhance = !need_light_enhance;
            }
        });
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }

    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        rgb = new Mat(height, width, CvType.CV_8UC3);
        ycrcb = new Mat(height, width, CvType.CV_8UC3);
        channels = new ArrayList<>(3);
        cr = new Mat(height, width, CvType.CV_8UC1);
        mask = new Mat(height, width, CvType.CV_8UC1);
        res = new Mat(height, width, CvType.CV_8UC4);
        count = 0;
        detect = new Mat(height, width, CvType.CV_8UC4);
        contours = new ArrayList<>();
    }

    public void onCameraViewStopped() {
        rgb.release();
        ycrcb.release();
        channels.clear();
        cr.release();
        mask.release();
        res.release();
        contours.clear();
    }
    public Mat light_enhance(Mat rgba,float fx,int code)
    {
        long begintime = System.currentTimeMillis();
        List<Mat> mv = new ArrayList<Mat>();
        Core.split(rgba,mv);
        Mat alpha = mv.get(3);
        mv.clear();

        int origin_h = rgba.rows(), origin_w = rgba.cols();
        Mat result = new Mat();
        Imgproc.resize(rgba,result,new org.opencv.core.Size(480.0/origin_h * origin_w ,480));
        int h = result.rows(),w = result.cols();
        float ratioMax = 7;
        Mat rgb = new Mat();
        Imgproc.cvtColor(result, rgb, code, 3);
        Mat imgScale = new Mat();
        rgb.convertTo(imgScale,CvType.CV_32FC3);
        Core.multiply(imgScale,new Scalar(1/255.0f,1/255.0f,1/255.0f,1),imgScale);
        Mat imgResize = new Mat();
        Imgproc.resize(imgScale,imgResize,new org.opencv.core.Size(0,0),fx,fx);
        List<Mat> channels = new ArrayList<>();

        Core.split(imgResize, channels);
        Mat t0 = new Mat();
        Core.max(channels.get(0),channels.get(1),t0);
        Core.max(t0,channels.get(2),t0);
        t0 = t0.t();
        int r = imgResize.rows(), c = imgResize.cols();
        Mat out = new Mat(c,r,CvType.CV_32FC1);
        long endtime = System.currentTimeMillis();
        Log.i(TAG,"Preprocess time: " + (endtime - begintime) + " ms");
        begintime = endtime;
        //TODO
        getK(t0.dataAddr(),out.dataAddr(),r,c);
        endtime = System.currentTimeMillis();
        Log.i(TAG,"Middle process time: " + (endtime - begintime) + " ms");
        begintime = endtime;

        Imgproc.resize(out.t(),out,new org.opencv.core.Size(w,h));
        Mat kRatio = new Mat();
        Core.divide(1.0,out,kRatio);
        Core.min(kRatio,new Scalar(ratioMax),kRatio);
        Core.pow(kRatio,arg_a,kRatio);

        mv.add(kRatio);
        mv.add(kRatio);
        mv.add(kRatio);
        Core.merge(mv,kRatio);
        Mat I=  new Mat(imgScale.size(), CvType.CV_32FC3);

        endtime = System.currentTimeMillis();
        Log.i(TAG,"Postprocess 1 time: " + (endtime - begintime) + " ms");
        begintime = endtime;

        matrixPow(imgScale.dataAddr(), kRatio.dataAddr(), I.dataAddr(),h*w*3);
        endtime = System.currentTimeMillis();
        Log.i(TAG,"Postprocess 2 time: " + (endtime - begintime) + " ms");
        begintime = endtime;

        //TODO
        Mat f = new Mat();
        Core.multiply(kRatio,new Scalar(-arg_b,-arg_b,-arg_b),f);
        Core.add(f,new Scalar(arg_b,arg_b,arg_b),f);
        Core.exp(f,f);
        Core.multiply(I,f,I);
        Core.multiply(I,new Scalar(255.0f,255.0f,255.0f),I);
        Core.min(I,new Scalar(255.0f,255.0f,255.0f),I);
        Core.max(I,new Scalar(0f,0f,0f),I);
        Mat En = new Mat();
        I.convertTo(En,CvType.CV_8UC3);
        Imgproc.resize(En,En,new org.opencv.core.Size(origin_w,origin_h),0,0,Imgproc.INTER_CUBIC);

        mv.clear();
        Core.split(En,mv);
        mv.add(alpha);
        Core.merge(mv,result);
        //copyRGB(En.dataAddr(),rgba.dataAddr(),h,w);
        //TODO Copy rgb
        endtime = System.currentTimeMillis();
        Log.i(TAG,"Postprocess 3 time: " + (endtime - begintime) + " ms");
        mv.clear();
        alpha.release();
        En.release();
        f.release();
        I.release();
        out.release();
        t0.release();
        rgb.release();
        imgResize.release();
        imgScale.release();
        return result;
    }
    public Mat ycrcb_otsu_detect(Mat rgba) {
        // get cr channel mask
        Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB, 3);
        Imgproc.cvtColor(rgb, ycrcb, Imgproc.COLOR_RGB2YCrCb, 3);
        Core.split(ycrcb, channels);
        cr = channels.get(1);
        Imgproc.threshold(cr, mask, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);

        // erode and dilate to denoise
        org.opencv.core.Size size = new org.opencv.core.Size(10, 10);
        Imgproc.erode(mask, mask, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, size));
        Imgproc.dilate(mask, mask, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, size));

        // keep max contour
//        ArrayList<MatOfPoint> contours = new ArrayList<>();
        contours.clear();
        Mat hierarchy = new Mat();
        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
        int max_index = 0;
        double max_size = -1;
        for (int i = 0; i < contours.size(); i++) {
            MatOfPoint ps = contours.get(i);
            //Log.i(TAG, "" + ps.size().height);
            if (ps.size().height > max_size) {
                max_size = ps.size().height;
                max_index = i;
            }
        }
        mask.setTo(new Scalar(0));
        Imgproc.drawContours(mask, contours, max_index, new Scalar(255), -1);
        res.setTo(new Scalar(0, 0, 0, 255));
        rgba.copyTo(res, mask);
        res.copyTo(rgba);
        Imgproc.cvtColor(rgba, detect, Imgproc.COLOR_RGBA2BGRA, 4);
        return rgba;
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        Mat rgba = inputFrame.rgba();
        long start = System.currentTimeMillis();
        rgba = ycrcb_otsu_detect(rgba);
        try {
            if(need_light_enhance)
                rgba = light_enhance(rgba,fx,Imgproc.COLOR_RGBA2RGB);
        }catch (Exception e)
        {
            Log.e("Error",e.getMessage());
        }

        long end = System.currentTimeMillis();
        Log.i(TAG, "Foreground Extraction Time: "+ (end - start) + "ms");
        //if(need_light_enhance)
        //    rgba = light_enhance(rgba);
        Imgproc.cvtColor(rgba, detect, Imgproc.COLOR_RGBA2BGRA, 4);
//        String fileName = Environment.getExternalStorageDirectory().getPath() +
//                "/hand_sample/0.jpg";
//
////        Utils.bitmapToMat(b,img);
//        Mat img = Imgcodecs.imread(fileName);
//        Mat rgba = new Mat();
//        Imgproc.cvtColor(img,rgba,Imgproc.COLOR_BGR2RGBA);
//        rgba = light_enhance(rgba);
        //rgba = ycrcb_otsu_detect(rgba);

        return rgba;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        List<String> effects = mOpenCvCameraView.getEffectList();

        if (effects == null) {
            Log.e(TAG, "Color effects are not supported by device!");
            return true;
        }

        mColorEffectsMenu = menu.addSubMenu("Color Effect");
        mEffectMenuItems = new MenuItem[effects.size()];

        int idx = 0;
        ListIterator<String> effectItr = effects.listIterator();
        while (effectItr.hasNext()) {
            String element = effectItr.next();
            mEffectMenuItems[idx] = mColorEffectsMenu.add(1, idx, Menu.NONE, element);
            idx++;
        }

        mResolutionMenu = menu.addSubMenu("Resolution");
        mResolutionList = mOpenCvCameraView.getResolutionList();
        mResolutionMenuItems = new MenuItem[mResolutionList.size()];

        ListIterator<Size> resolutionItr = mResolutionList.listIterator();
        idx = 0;
        while (resolutionItr.hasNext()) {
            Size element = resolutionItr.next();
            mResolutionMenuItems[idx] = mResolutionMenu.add(2, idx, Menu.NONE,
                    Integer.valueOf(element.width).toString() + "x" + Integer.valueOf(element.height).toString());
            idx++;
        }

        return true;
    }

    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item.getGroupId() == 1) {
            mOpenCvCameraView.setEffect((String) item.getTitle());
            Toast.makeText(this, mOpenCvCameraView.getEffect(), Toast.LENGTH_SHORT).show();
        } else if (item.getGroupId() == 2) {
            int id = item.getItemId();
            Size resolution = mResolutionList.get(id);
            mOpenCvCameraView.setResolution(resolution);
            resolution = mOpenCvCameraView.getResolution();
            String caption = Integer.valueOf(resolution.width).toString() + "x" + Integer.valueOf(resolution.height).toString();
            Toast.makeText(this, caption, Toast.LENGTH_SHORT).show();
        }

        return true;
    }

    @SuppressLint("SimpleDateFormat")
    @Override
    public boolean onTouch(View v, MotionEvent event) {
        Log.i(TAG, "onTouch event");
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
        String currentDateandTime = sdf.format(new Date());
        String fileName_pre = Environment.getExternalStorageDirectory().getPath() +
                "/hand_sample/sample_picture_" + currentDateandTime;
//        mOpenCvCameraView.takePicture(fileName);
        Imgcodecs.imwrite(fileName_pre +".png", detect);
        for(int  fx = 128;fx >= 2;fx >>= 1){
            long begintime = System.currentTimeMillis();
            Mat detect_en = light_enhance(detect,1.0f/fx,Imgproc.COLOR_BGRA2BGR);
            long endtime = System.currentTimeMillis();
            StringBuilder msq = new StringBuilder("Light enhancement with resolution: (");
            msq.append(detect.rows());
            msq.append(", ");
            msq.append(detect.cols());
            msq.append("),/");
            msq.append(fx);
            msq.append(", Time: ");
            msq.append(endtime - begintime);
            msq.append("ms");
            Log.i(TAG, msq.toString());
            Imgcodecs.imwrite(fileName_pre +"_en_" + fx + ".png", detect_en);
        }
        Toast.makeText(this, fileName_pre + " saved", Toast.LENGTH_SHORT).show();
        return false;
    }

    public void dispatch_directory_choose(View view)
    {
        return;
    }
}

