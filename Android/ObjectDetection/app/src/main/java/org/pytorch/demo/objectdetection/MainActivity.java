// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package org.pytorch.demo.objectdetection;

import android.Manifest;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.StringReader;
import java.nio.FloatBuffer;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class MainActivity extends AppCompatActivity implements Runnable {

//static {
//    if (!NativeLoader.isInitialized()) {
//        NativeLoader.init(new SystemDelegate());
//    }
//    NativeLoader.loadLibrary("pytorch_jni");
//    NativeLoader.loadLibrary("torchvision_ops");
//}

    private int mImageIndex = 0;
    private String[] mTestImages = {"test1.jpg", "test2.jpg", "test3.png", "test4.jpg"};

    private ImageView mImageView;
    private ResultView mResultView;
    private Button mButtonDetect;
    private ProgressBar mProgressBar;
    private Bitmap mBitmap = null;
    private Module mModule = null;
    private float mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY;

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 1);
        }

        setContentView(R.layout.activity_main);

        try {
            mBitmap = BitmapFactory.decodeStream(getAssets().open(mTestImages[mImageIndex]));
        } catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            finish();
        }

        mImageView = findViewById(R.id.imageView);
        mImageView.setImageBitmap(mBitmap);
        mResultView = findViewById(R.id.resultView);
        mResultView.setVisibility(View.INVISIBLE);

        final Button buttonTest = findViewById(R.id.testButton);
        buttonTest.setText(("Test Image 1/4"));
        buttonTest.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mResultView.setVisibility(View.INVISIBLE);
                mImageIndex = (mImageIndex + 1) % mTestImages.length;
                buttonTest.setText(String.format("Text Image %d/%d", mImageIndex + 1, mTestImages.length));

                try {
                    mBitmap = BitmapFactory.decodeStream(getAssets().open(mTestImages[mImageIndex]));
                    mImageView.setImageBitmap(mBitmap);
                } catch (IOException e) {
                    Log.e("Object Detection", "Error reading assets", e);
                    finish();
                }
            }
        });


        final Button buttonSelect = findViewById(R.id.selectButton);
        buttonSelect.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mResultView.setVisibility(View.INVISIBLE);

                final CharSequence[] options = { "Choose from Photos", "Take Picture", "Cancel" };
                AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
                builder.setTitle("New Test Image");

                builder.setItems(options, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int item) {
                        if (options[item].equals("Take Picture")) {
                            Intent takePicture = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                            startActivityForResult(takePicture, 0);
                        }
                        else if (options[item].equals("Choose from Photos")) {
                            Intent pickPhoto = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.INTERNAL_CONTENT_URI);
                            startActivityForResult(pickPhoto , 1);
                        }
                        else if (options[item].equals("Cancel")) {
                            dialog.dismiss();
                        }
                    }
                });
                builder.show();
            }
        });

        final Button buttonLive = findViewById(R.id.liveButton);
        buttonLive.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
              final Intent intent = new Intent(MainActivity.this, ObjectDetectionActivity.class);
              startActivity(intent);
            }
        });

        mButtonDetect = findViewById(R.id.detectButton);
        mProgressBar = (ProgressBar) findViewById(R.id.progressBar);
        mButtonDetect.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mButtonDetect.setEnabled(false);
                mProgressBar.setVisibility(ProgressBar.VISIBLE);
                mButtonDetect.setText(getString(R.string.run_model));

                mImgScaleX = (float)mBitmap.getWidth() / PrePostProcessor.INPUT_WIDTH;
                mImgScaleY = (float)mBitmap.getHeight() / PrePostProcessor.INPUT_HEIGHT;

                mIvScaleX = (mBitmap.getWidth() > mBitmap.getHeight() ? (float)mImageView.getWidth() / mBitmap.getWidth() : (float)mImageView.getHeight() / mBitmap.getHeight());
                mIvScaleY  = (mBitmap.getHeight() > mBitmap.getWidth() ? (float)mImageView.getHeight() / mBitmap.getHeight() : (float)mImageView.getWidth() / mBitmap.getWidth());

                mStartX = (mImageView.getWidth() - mIvScaleX * mBitmap.getWidth())/2;
                mStartY = (mImageView.getHeight() -  mIvScaleY * mBitmap.getHeight())/2;
                System.out.println("mImgScaleX: " + mImgScaleX);
                System.out.println("mImgScaleY: " + mImgScaleY);
                System.out.println("mIvScaleX: " + mIvScaleX);
                System.out.println("mIvScaleY: " + mIvScaleY);
                System.out.println("mStartX: " + mStartX);
                System.out.println("mStartY: " + mStartY);
                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });

        try {
            mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "ssd300_pruned_optimized.torchscript.ptl"));
            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open("classes.txt")));
            String line;
            List<String> classes = new ArrayList<>();
            while ((line = br.readLine()) != null) {
                classes.add(line);
            }
            PrePostProcessor.mClasses = new String[classes.size()];
            classes.toArray(PrePostProcessor.mClasses);
        } catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            finish();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_CANCELED) {
            switch (requestCode) {
                case 0:
                    if (resultCode == RESULT_OK && data != null) {
                        mBitmap = (Bitmap) data.getExtras().get("data");
                        Matrix matrix = new Matrix();
//                        matrix.postRotate(90.0f);
                        mBitmap = Bitmap.createBitmap(mBitmap, 0, 0, mBitmap.getWidth(), mBitmap.getHeight(), matrix, true);
                        mImageView.setImageBitmap(mBitmap);
                    }
                    break;
                case 1:
                    if (resultCode == RESULT_OK && data != null) {
                        Uri selectedImage = data.getData();
                        String[] filePathColumn = {MediaStore.Images.Media.DATA};
                        if (selectedImage != null) {
                            Cursor cursor = getContentResolver().query(selectedImage,
                                    filePathColumn, null, null, null);
                            if (cursor != null) {
                                cursor.moveToFirst();
                                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                                String picturePath = cursor.getString(columnIndex);
                                mBitmap = BitmapFactory.decodeFile(picturePath);
                                Matrix matrix = new Matrix();
//                                matrix.postRotate(90.0f);
                                mBitmap = Bitmap.createBitmap(mBitmap, 0, 0, mBitmap.getWidth(), mBitmap.getHeight(), matrix, true);
                                mImageView.setImageBitmap(mBitmap);
                                cursor.close();
                            }
                        }
                    }
                    break;
            }
        }
    }

    @Override
    public void run() {
//        Bitmap resizedBitmap = Bitmap.createScaledBitmap(mBitmap, PrePostProcessor.INPUT_WIDTH, PrePostProcessor.INPUT_HEIGHT, true);
//
//        final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * resizedBitmap.getWidth() * resizedBitmap.getHeight());
//        TensorImageUtils.bitmapToFloatBuffer(resizedBitmap, 0,0,resizedBitmap.getWidth(),resizedBitmap.getHeight(), PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB, floatBuffer, 0);
//        final Tensor inputTensor =  Tensor.fromBlob(floatBuffer, new long[] {3, resizedBitmap.getHeight(), resizedBitmap.getWidth()});
//
//        final long startTime = SystemClock.elapsedRealtime();
//
////        System.out.println("Type: " + );
//        IValue outputTuple = mModule.forward(IValue.from(IValue.from(inputTensor).toTensor())); //.toTuple(); //IValue.listFrom(
////        System.out.println("Type: " + outputTuple);
//        final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
//        Log.d("D2Go",  "inference time (ms): " + inferenceTime);
//
//
//        final Tensor outputTensor = outputTuple.toTensor();
//        final float[] outputData = outputTensor.getDataAsFloatArray();
//
//        final int numBatch = (int) outputTensor.shape()[0];
//        final int numClasses = (int) outputTensor.shape()[1];
//        final int topK = (int) outputTensor.shape()[2];
//        final int outputColumn = 6;
//
//        final ArrayList<Result> results = new ArrayList<>();
//
//        for (int i = 0; i < numBatch; i++) {
//            float imgScaleX = (float) mImageView.getWidth() / PrePostProcessor.INPUT_WIDTH;
//            float imgScaleY = (float) mImageView.getHeight() / PrePostProcessor.INPUT_HEIGHT;
//            float ivScaleX = (float) mImageView.getWidth() / PrePostProcessor.INPUT_WIDTH;
//            float ivScaleY = (float) mImageView.getHeight() / PrePostProcessor.INPUT_HEIGHT;
//            float startX = mImageView.getX();
//            float startY = mImageView.getY();
//
//            int countResult = 0;
//            for (int cl = 0; cl < numClasses; cl++) {
//                for (int k = 0; k < topK; k++) {
//                    float score = outputData[i * (numClasses * topK * outputColumn) + cl * (topK * outputColumn) + k * outputColumn + 4];
//                    if (score < 0.01) { // confidence threshold
//                        break;
//                    }
//
//                    float left = outputData[i * (numClasses * topK * outputColumn) + cl * (topK * outputColumn) + k * outputColumn];
//                    float top = outputData[i * (numClasses * topK * outputColumn) + cl * (topK * outputColumn) + k * outputColumn + 1];
//                    float right = outputData[i * (numClasses * topK * outputColumn) + cl * (topK * outputColumn) + k * outputColumn + 2];
//                    float bottom = outputData[i * (numClasses * topK * outputColumn) + cl * (topK * outputColumn) + k * outputColumn + 3];
//
//                    left = imgScaleX * left;
//                    top = imgScaleY * top;
//                    right = imgScaleX * right;
//                    bottom = imgScaleY * bottom;
//
//                    Rect rect = new Rect((int) (startX + ivScaleX * left), (int) (startY + top * ivScaleY), (int) (startX + ivScaleX * right), (int) (startY + ivScaleY * bottom));
//                    Result result = new Result(cl, score, rect);
//                    results.add(result);
//                    countResult++;
//
//                    if (countResult >= topK) {
//                        break;
//                    }
//                }
//            }
//        }
//
//        runOnUiThread(() -> {
//            mButtonDetect.setEnabled(true);
//            mButtonDetect.setText(getString(R.string.detect));
//            mProgressBar.setVisibility(ProgressBar.INVISIBLE);
//            mResultView.setResults(results);
//            mResultView.invalidate();
//            mResultView.setVisibility(View.VISIBLE);
//        });
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(mBitmap, PrePostProcessor.INPUT_WIDTH, PrePostProcessor.INPUT_HEIGHT, true);

//        final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * resizedBitmap.getWidth() * resizedBitmap.getHeight());
//        TensorImageUtils.bitmapToFloatBuffer(resizedBitmap, 0,0,resizedBitmap.getWidth(),resizedBitmap.getHeight(), PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB, floatBuffer, 0);
//        final Tensor inputTensor =  Tensor.fromBlob(floatBuffer, new long[] {1, 3, resizedBitmap.getHeight(), resizedBitmap.getWidth()});
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap, PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB);


        System.out.println("output: " + Arrays.toString(inputTensor.shape()));
        final long startTime = SystemClock.elapsedRealtime();

        final IValue[] outputTuple  = mModule.forward(IValue.from(inputTensor)).toTuple();
        final Tensor detections = outputTuple[0].toTensor();
//        System.out.println("output: " + detections);
        final float[] outputArray = detections.getDataAsFloatArray();
//        for(int i=0; i< 21000; i++){
//            if(outputArray[i] != 0.0) {
//                System.out.println("outputArray: " + i + " "  + outputArray[i]);
//            }
//        }
        final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
        Log.d("D2Go",  "inference time (ms): " + inferenceTime);
        int size1 = (int) detections.shape()[1];
        int size2 = (int) detections.shape()[2];
        int size3 = (int) detections.shape()[3];
        System.out.println("output: " + outputArray.length);
        System.out.println("Size1: " + size1);
        System.out.println("Size2: " + size2);
        System.out.println("Size3: " + size3);
        float[] outputs = new float[200];
        int count = 0;
        for (int i = 0; i < size1; i++) {
            int j = 0;
//            System.out.println("scores: " + outputArray[i * size2 * size3 + j * size3]);
            while (outputArray[i * size2 * size3 + j * size3] >= 0.035) {     //outputArray[i * size2 * size3 * 4 + j * 4
                System.out.println("outputArray number: " + (i * size2 * size3 + j * size3));
                float score = outputArray[i * size2 * size3 + j * size3];
//                String labelName = PrePostProcessor.mClasses[i-1];
//                String displayTxt = String.format("%s: %.2f", labelName, score);
                float[] pt = new float[4];
                for (int k = 0; k < 4; k++) {
                    pt[k] = outputArray[i * size2 * size3 + j * size3 + k + 1];
                }
                for (int k = 0; k < 4; k++) {
                    System.out.println("pt: "+ k + " " + pt[k]);
                }
//                float[] coords = {pt[0], pt[1], pt[2] - pt[0] + 1, pt[3] - pt[1] + 1};
//                float[] coords = {pt[0], pt[3], pt[2], pt[1]};
//                System.out.println("coords: " + coords);
//                Color color = colors[i];
                // Add your code to use `coords`, `color`, and `displayTxt` here.
                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 0] = pt[0] * PrePostProcessor.INPUT_WIDTH; //coords[0];
                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 1] = pt[3] * PrePostProcessor.INPUT_HEIGHT; //coords[1];
                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 2] = pt[2] * PrePostProcessor.INPUT_WIDTH; //coords[2];
                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 3] = pt[1] * PrePostProcessor.INPUT_WIDTH; //coords[3];
                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 4] = score;
                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 5] = i-1;
                j++;
                count++;
            }
        }
//        for (int i =0; i < 100; i++){
//            System.out.println("outputs: " + i + " " + outputs[i]);
//        }
//        System.out.println("count: " + count);
        System.out.println("bitmap width: " + mBitmap.getWidth());
        System.out.println("bitmap height: " + mBitmap.getHeight());
        System.out.println("imgview width: " + mImageView.getWidth());
        System.out.println("imgview height: " + mImageView.getHeight());
        final ArrayList<Result> results = PrePostProcessor.outputsToPredictions(count, outputs, mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY);

        runOnUiThread(() -> {
            mButtonDetect.setEnabled(true);
            mButtonDetect.setText(getString(R.string.detect));
            mProgressBar.setVisibility(ProgressBar.INVISIBLE);
            mResultView.setResults(results);
            mResultView.invalidate();
            mResultView.setVisibility(View.VISIBLE);
        });



//        final long startTime = SystemClock.elapsedRealtime();
//        IValue[] outputTuple = mModule.forward(IValue.listFrom(inputTensor)).toTuple();
//        final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
//        Log.d("D2Go",  "inference time (ms): " + inferenceTime);
//        final Map<String, IValue> map = outputTuple[1].toList()[0].toDictStringKey();
//        float[] boxesData = new float[]{};
//        float[] scoresData = new float[]{};
//        long[] labelsData = new long[]{};
//        if (map.containsKey("boxes")) {
//            final Tensor boxesTensor = map.get("boxes").toTensor();
//            final Tensor scoresTensor = map.get("scores").toTensor();
//            final Tensor labelsTensor = map.get("labels").toTensor();
//            boxesData = boxesTensor.getDataAsFloatArray();
//            scoresData = scoresTensor.getDataAsFloatArray();
//            labelsData = labelsTensor.getDataAsLongArray();
//
//            final int n = scoresData.length;
//            float[] outputs = new float[n * PrePostProcessor.OUTPUT_COLUMN];
//            int count = 0;
//            for (int i = 0; i < n; i++) {
////                if (scoresData[i] < 0.5)
////                    continue;
//
//                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 0] = boxesData[4 * i + 0];
//                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 1] = boxesData[4 * i + 1];
//                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 2] = boxesData[4 * i + 2];
//                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 3] = boxesData[4 * i + 3];
//                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 4] = scoresData[i];
//                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 5] = labelsData[i] - 1;
//                count++;
//            }
//
//            final ArrayList<Result> results = PrePostProcessor.outputsToPredictions(count, outputs, mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY);
//
//            runOnUiThread(() -> {
//                mButtonDetect.setEnabled(true);
//                mButtonDetect.setText(getString(R.string.detect));
//                mProgressBar.setVisibility(ProgressBar.INVISIBLE);
//                mResultView.setResults(results);
//                mResultView.invalidate();
//                mResultView.setVisibility(View.VISIBLE);
//            });
//        }
    }
}
