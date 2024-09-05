package org.pytorch.demo.objectdetection;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.PostProcessor;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.SystemClock;
import android.util.Log;
import android.view.TextureView;
import android.view.ViewStub;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Map;

public class ObjectDetectionActivity extends AbstractCameraXActivity<ObjectDetectionActivity.AnalysisResult> {
    private Module mModule = null;
    private ResultView mResultView;

    static class AnalysisResult {
        private final ArrayList<Result> mResults;

        public AnalysisResult(ArrayList<Result> results) {
            mResults = results;
        }
    }

    @Override
    protected int getContentViewLayoutId() {
        return R.layout.activity_object_detection;
    }

    @Override
    protected TextureView getCameraPreviewTextureView() {
        mResultView = findViewById(R.id.resultView);
        return ((ViewStub) findViewById(R.id.object_detection_texture_view_stub))
                .inflate()
                .findViewById(R.id.object_detection_texture_view);
    }

    @Override
    protected void applyToUiAnalyzeImageResult(AnalysisResult result) {
        mResultView.setResults(result.mResults);
        mResultView.invalidate();
    }

    private Bitmap imgToBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    @Override
    @WorkerThread
    @Nullable
    protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {
        if (mModule == null) {
//            mModule = PyTorchAndroid.loadModuleFromAsset(getAssets(), "d2go.pt");
//            mModule = PyTorchAndroid.loadModuleFromAsset(getAssets(), "ssd300_100.pt");
            try {
                mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "ssd300_pruned_optimized.torchscript.ptl"));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        Bitmap bitmap =  imgToBitmap(image.getImage());
        Matrix matrix = new Matrix();
//        matrix.postRotate(90.0f);
        bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, PrePostProcessor.INPUT_WIDTH, PrePostProcessor.INPUT_HEIGHT, true);

        final FloatBuffer floatBuffer = Tensor.allocateFloatBuffer(3 * bitmap.getWidth() * bitmap.getHeight());
//        TensorImageUtils.bitmapToFloatBuffer(bitmap, 0,0,bitmap.getWidth(),bitmap.getHeight(), PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB, floatBuffer, 0);
//        final Tensor inputTensor =  Tensor.fromBlob(floatBuffer, new long[] {1, 3, bitmap.getHeight(), bitmap.getWidth()});
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap, PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB);
//        Tensor detections  = mModule.forward(IValue.from(inputTensor)).toTensor();
        final IValue[] outputTuple  = mModule.forward(IValue.from(inputTensor)).toTuple();
        final Tensor detections = outputTuple[0].toTensor();
        float[] outputArray = detections.getDataAsFloatArray();


        int size1 = (int) detections.shape()[1];
        int size2 = (int) detections.shape()[2];
        int size3 = (int) detections.shape()[3];
        float[] outputs = new float[200];
        int count = 0;
        for (int i = 0; i < size1; i++) {
            int j = 0;
            while (outputArray[i * size2 * size3 + j * size3] >= 0.025) {     //outputArray[i * size2 * size3 * 4 + j * 4
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
                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 0] = pt[0]*PrePostProcessor.INPUT_WIDTH; //coords[0];
                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 1] = pt[3]*PrePostProcessor.INPUT_HEIGHT; //coords[1];
                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 2] = pt[2]*PrePostProcessor.INPUT_WIDTH; //coords[2];
                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 3] = pt[1]*PrePostProcessor.INPUT_HEIGHT; //coords[3];
                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 4] = score;
                outputs[PrePostProcessor.OUTPUT_COLUMN * count + 5] = i-1;
                j++;
                count++;
            }
        }
        System.out.println("bitmap width: " + bitmap.getWidth());
        System.out.println("bitmap height: " + bitmap.getHeight());
        System.out.println("resultview width: " +  mResultView.getWidth());
        System.out.println("resultview height: " +  mResultView.getHeight());

        float imgScaleX = (float) bitmap.getWidth() / PrePostProcessor.INPUT_WIDTH;
        float imgScaleY = (float) bitmap.getHeight() / PrePostProcessor.INPUT_HEIGHT;
        float ivScaleX = (float) mResultView.getWidth() / bitmap.getWidth();//(bitmap.getWidth() > bitmap.getHeight() ? (float)mResultView.getWidth() / bitmap.getWidth() : (float)mResultView.getHeight() / bitmap.getHeight());//
        float ivScaleY = (float) mResultView.getHeight() / bitmap.getHeight();//(bitmap.getHeight() > bitmap.getWidth() ? (float)mResultView.getHeight() / bitmap.getHeight() : (float)mResultView.getWidth() / bitmap.getWidth());//
        final ArrayList<Result> results = PrePostProcessor.outputsToPredictions(count, outputs, imgScaleX, imgScaleY, ivScaleX, ivScaleY, 0, 0);
        return new AnalysisResult(results);
//        IValue[] outputTuple = mModule.forward(IValue.listFrom(inputTensor)).toTuple();
//        final Map<String, IValue> map = outputTuple[1].toList()[0].toDictStringKey();
//        float[] boxesData = new float[]{};
//        float[] scoresData = new float[]{};
//        long[] labelsData = new long[]{};
//
//        if (map.containsKey("boxes")) {
//            final Tensor boxesTensor = map.get("boxes").toTensor();
//            final Tensor scoresTensor = map.get("scores").toTensor();
//            final Tensor labelsTensor = map.get("labels").toTensor();
//            boxesData = boxesTensor.getDataAsFloatArray();
//            scoresData = scoresTensor.getDataAsFloatArray();
//            labelsData = labelsTensor.getDataAsLongArray();
//
//            final int n = scoresData.length;
//            int count = 0;
//            float[] outputs = new float[n * PrePostProcessor.OUTPUT_COLUMN];
//            for (int i = 0; i < n; i++) {
////                if (scoresData[i] < 0.3)
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
//            float imgScaleX = (float) bitmap.getWidth() / PrePostProcessor.INPUT_WIDTH;
//            float imgScaleY = (float) bitmap.getHeight() / PrePostProcessor.INPUT_HEIGHT;
//            float ivScaleX = (float) mResultView.getWidth() / bitmap.getWidth();
//            float ivScaleY = (float) mResultView.getHeight() / bitmap.getHeight();
//
//            final ArrayList<Result> results = PrePostProcessor.outputsToPredictions(count, outputs, imgScaleX, imgScaleY, ivScaleX, ivScaleY, 0, 0);
//            return new AnalysisResult(results);
//        }
//        return null;
    }
}
