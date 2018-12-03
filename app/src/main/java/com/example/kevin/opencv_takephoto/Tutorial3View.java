package com.example.kevin.opencv_takephoto;

import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;

import org.opencv.android.JavaCameraView;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.Camera;
import android.hardware.Camera.PictureCallback;
import android.hardware.Camera.Size;
import android.util.AttributeSet;
import android.util.Log;

public class Tutorial3View extends JavaCameraView implements PictureCallback {

    private static final String TAG = "Sample::Tutorial3View";
    private String mPictureFileName;

    public Tutorial3View(Context context, AttributeSet attrs) {
        super(context, attrs);
    }


    public List<String> getEffectList() {
        return mCamera.getParameters().getSupportedColorEffects();
    }

    public boolean isEffectSupported() {
        return (mCamera.getParameters().getColorEffect() != null);
    }

    public String getEffect() {
        return mCamera.getParameters().getColorEffect();
    }

    public void setEffect(String effect) {
        Camera.Parameters params = mCamera.getParameters();
        params.setColorEffect(effect);
        mCamera.setParameters(params);
    }

    public List<Size> getResolutionList() {
        return mCamera.getParameters().getSupportedPreviewSizes();
    }

    public void setResolution(Size resolution) {
        disconnectCamera();
        mMaxHeight = resolution.height;
        mMaxWidth = resolution.width;
        connectCamera(getWidth(), getHeight());
    }

    public Size getResolution() {
        return mCamera.getParameters().getPreviewSize();
    }

    public void takePicture(final String fileName) {
        Log.i(TAG, "Taking picture");
        this.mPictureFileName = fileName;
        // Postview and jpeg are sent in the same buffers if the queue is not empty when performing a capture.
        // Clear up buffers to avoid mCamera.takePicture to be stuck because of a memory issue
        mCamera.setPreviewCallback(null);
        // PictureCallback is implemented by the current class
        mCamera.takePicture(null, null, this);
    }

    public Mat ycrcb_otsu_detect(Mat rgba)
    {
        Mat rgb = new Mat();
        Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB, 3);
        Mat ycrcb = new Mat();
        Imgproc.cvtColor(rgb, ycrcb, Imgproc.COLOR_RGB2YCrCb, 3);
        ArrayList<Mat> channels = new ArrayList<>(3);
        Core.split(ycrcb, channels);
        Mat cr  = channels.get(1);
        Mat mask = new Mat();
        Log.i(TAG, "mask1: " + mask.size());
        Imgproc.threshold(cr, mask, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
        Log.i(TAG, "mask2: " + mask.size());
        Mat res = new Mat(rgba.height(), rgba.width(), rgba.type(), new Scalar(0, 0, 0, 255));
        rgba.copyTo(res, mask);

        return res;
    }

    @Override
    public void onPictureTaken(byte[] data, Camera camera) {
        Log.i(TAG, "Saving a bitmap to file");
        // The camera preview was automatically stopped. Start it again.
        mCamera.startPreview();
        mCamera.setPreviewCallback(this);

        Bitmap bmp = BitmapFactory.decodeByteArray(data, 0, data.length);
        Mat img = new Mat(bmp.getHeight(), bmp.getWidth(), CvType.CV_8UC3);
        Bitmap bmp32 = bmp.copy(Bitmap.Config.ARGB_8888, true);
        Utils.bitmapToMat(bmp32, img);
        Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2RGB, 4);

//        img = ycrcb_otsu_detect(img);
        Imgcodecs.imwrite(mPictureFileName, img);
        return;
//        Imgproc.cvtColor(img, img, Imgproc.COLOR_RGBA2BGRA, 4);
//        Utils.matToBitmap(img, bmp);
//        ByteArrayOutputStream stream = new ByteArrayOutputStream();
//        bmp.compress(Bitmap.CompressFormat.PNG, 100, stream);
//        byte[] processed_data = stream.toByteArray();
//        bmp.recycle();
//
//        // Write the image in a file (in jpeg format)
//        try {
//            FileOutputStream fos = new FileOutputStream(mPictureFileName);
//
//            fos.write(processed_data);
//            fos.close();
//
//        } catch (java.io.IOException e) {
//            Log.e("PictureDemo", "Exception in photoCallback", e);
//        }

    }
}

