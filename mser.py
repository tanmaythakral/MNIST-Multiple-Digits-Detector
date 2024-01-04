import cv2
import numpy as np

class MSERProcessor:
    def __init__(self, image_path, similarity_threshold=0.1):
        self.image = cv2.imread(image_path, 0)
        self.mser = cv2.MSER_create()
        self.similarity_threshold = similarity_threshold

    def process_image(self):
        regions, _ = self.mser.detectRegions(self.image)
        boxes = [cv2.boundingRect(region) for region in regions]
        larger_boxes = self._get_larger_boxes(boxes)
        mean_boxes = self._get_mean_boxes(set(larger_boxes))

        for box in mean_boxes:
            x, y, w, h = box
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        print(len(mean_boxes))
        print(mean_boxes)
        cv2.imshow('MSER', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return mean_boxes

    def _get_larger_boxes(self, boxes):
        boxes.sort(key=lambda box: box[2]*box[3], reverse=True)

        output_boxes = []
        for i, box1 in enumerate(boxes):
            x1, y1, w1, h1 = box1
            is_inside_another_box = False
            for box2 in boxes[:i]:
                x2, y2, w2, h2 = box2
                if x1 > x2 and y1 > y2 and x1 + w1 < x2 + w2 and y1 + h1 < y2 + h2:
                    is_inside_another_box = True
                    break
            if not is_inside_another_box:
                output_boxes.append(box1)
        return output_boxes

    def _calculate_iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        intersection_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        union_area = w1 * h1 + w2 * h2 - intersection_area

        iou = intersection_area / union_area if union_area > 0 else 0
        return iou

    def _get_mean_boxes(self, boxes):
        mean_boxes = set()

        for box in boxes:
            x, y, w, h = box
            box_center = (x + w / 2, y + h / 2)

            similar_boxes = [box]
            for other_box in boxes:
                if box != other_box and self._calculate_iou(box, other_box) > self.similarity_threshold:
                    similar_boxes.append(other_box)

            # Calculate the mean of similar boxes
            if similar_boxes:
                mean_box = np.mean(similar_boxes, axis=0).astype(int)
                mean_boxes.add(tuple(mean_box))

        return mean_boxes

# Example usage
if __name__ == "__main__":
    mser_processor = MSERProcessor('test_images/images9.png', similarity_threshold=0.1)
    mser_processor.process_image()
