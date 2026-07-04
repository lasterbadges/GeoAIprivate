import cv2
import numpy
import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNetUNet(nn.Module):
    def __init__(self, freeze=True):
        super(ResNetUNet, self).__init__()

        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.left_conv0 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu) # 64 канала
        self.left_pool = base_model.maxpool
        self.left_conv1 = base_model.layer1 # 64 канала
        self.left_conv2 = base_model.layer2 # 128 каналов
        self.left_conv3 = base_model.layer3 # 256 каналов
        self.left_conv4 = base_model.layer4 # 512 каналов (bottleneck)

        if freeze:
            for param in [self.left_conv0, self.left_conv1, self.left_conv2, self.left_conv3, self.left_conv4]:
                for p in param.parameters():
                    p.requires_grad = False

        self.right_conv1 = nn.Sequential(
            nn.Conv2d(768, 256, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, (3, 3), padding=1),
            nn.ReLU(),
        )
        
        self.right_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, (3, 3), padding=1),
            nn.ReLU()
        )

        self.right_conv3 = nn.Sequential(
            nn.Conv2d(128, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, (3, 3), padding=1),
            nn.ReLU()
        )

        self.right_conv4 = nn.Sequential(
            nn.Conv2d(96, 32, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, (3, 3), padding=1),
            nn.Sigmoid()
        )

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        left0 = self.left_conv0(x) # (H/2 x W/2 x 64)
        x = self.left_pool(left0) # (H/4 x W/4 x 64)
        left1 = self.left_conv1(x) # (H/4 x W/4 x 64)
        left2 = self.left_conv2(left1) # (H/8 x W/8 x 128)
        left3 = self.left_conv3(left2) # (H/16 x W/16 x 256)
        left4 = self.left_conv4(left3) # (H/32 x W/32 x 512)

        up1 = self.up(left4) # (H/16 x W/16 x 512)
        concat1 = torch.cat((up1, left3), 1) # (H/16 x W/16 x 768)
        right1 = self.right_conv1(concat1) # (H/16 x W/16 x 128)

        up2 = self.up(right1) # (H/8 x H/8 x 128)
        concat2 = torch.cat((up2, left2), 1) # (H/8 x H/8 x 256)
        right2 = self.right_conv2(concat2) # (H/8 x H/8 x 64)

        up3 = self.up(right2) # (H/4 x H/4 x 64)
        concat3 = torch.cat((up3, left1), 1) # (H/4 x H/4 x 128)
        right3 = self.right_conv3(concat3) # (H/4 x H/4 x 32)

        up4 = self.up(right3) # (H/2 x H/2 x 32)
        concat4 = torch.cat((up4, left0), 1) # (H/2 x H/2 x 96)
        right4 = self.right_conv4(concat4) # (H/2, W/2, 1)

        output = self.up(right4) # (H x W x 1)
        return output

class PreProcessingMethods:
    def __init__(self):
        self.unet_model = None
        self.resnet_model = None

    def is_grayscale_like(self, image_bgr):
        """Определяет, является ли изображение фактически черно-белым."""
        if image_bgr.ndim == 3:
            ch_std = numpy.std(image_bgr.reshape(-1, 3).astype(numpy.float32), axis=0)
            return bool(numpy.max(ch_std) < 8)
        return True

    def normalize_for_inference(self, image_bgr):
        """
        Нормализует изображение перед подачей в нейросети:
        - ч/б (1-канальный / однотонный BGR) -> полноцветный RGB-эквивалент
        - выравнивание экспозиции через CLAHE в LAB
        - гамма-коррекция под среднюю яркость
        - адаптивный white-balance
        Возвращает BGR uint8 того же размера, пригодный для нейросетей.
        """
        img = image_bgr.copy()

        # определяем, является ли изображение фактически ч/б:
        # если std по каналам < 5 — все каналы почти одинаковые
        if img.ndim == 3:
            ch_std = numpy.std(img.reshape(-1, 3).astype(numpy.float32), axis=0)
            is_grayscale = bool(numpy.max(ch_std) < 8)
        else:
            is_grayscale = True

        if is_grayscale:
            # конвертируем в серый, затем обратно в псевдо-RGB
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            # CLAHE на сером
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray_eq = clahe.apply(gray)
            img = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
        else:
            # цветное изображение: выравниваем экспозицию в LAB-пространстве
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_ch = clahe.apply(l_ch)
            img = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

            # простой серого мира white balance
            b_m, g_m, r_m = [numpy.mean(img[:, :, i]) + 1e-6 for i in range(3)]
            gray_mean = (b_m + g_m + r_m) / 3.0
            img = numpy.clip(
                img.astype(numpy.float32) * [gray_mean / b_m, gray_mean / g_m, gray_mean / r_m],
                0, 255
            ).astype(numpy.uint8)

        # гамма-коррекция: подтягиваем темные снимки
        mean_lum = numpy.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if mean_lum < 80:
            gamma = 0.6  # осветляем
        elif mean_lum > 200:
            gamma = 1.5  # притемняем
        else:
            gamma = 1.0

        if gamma != 1.0:
            lut = numpy.array([min(255, int(((i / 255.0) ** gamma) * 255)) for i in range(256)], dtype=numpy.uint8)
            img = cv2.LUT(img, lut)

        return img

    # Подготовка изображеняи цветокорекция clahe возваращает три копии в разных цвет пространствах
    def _ensure_color(self, image_bgr):
        """если снимок фактически ч/б — разворачиваем gray→BGR без изменения пикселей."""
        if image_bgr.ndim == 3:
            ch_std = numpy.std(image_bgr.reshape(-1, 3).astype(numpy.float32), axis=0)
            if numpy.max(ch_std) < 8:  # все каналы почти одинаковые → ч/б
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return image_bgr

    def preprocess_image(self, image_path, max_side=3200):
        # 1. Загрузка изображения в формате BGR
        image_array = numpy.fromfile(str(image_path), dtype=numpy.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение по пути: {image_path}")

        # 1.5. Автоматический быстрый ресайз для больших панорам
        h, w = image_bgr.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            new_w = int(w * scale)
            new_h = int(h * scale)
            image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 2. Перевод в оттенки серого
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # 3. Настройка и применение адаптивного выравнивания контраста (CLAHE)
        # clipLimit=3.0 — порог ограничения контраста. Чем выше, тем контрастнее (и больше шума).
        # tileGridSize=(8, 8) — размер сетки, на которую бьется картинка для локального анализа.
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        image_enhanced_gray = clahe.apply(image_gray)

        # 4. Перевод оригинального BGR в HSV для последующего выделения синих контуров
        # cv2.COLOR_BGR2HSV меняет представление с каналов B-G-R на Тон-Насыщенность-Яркость.
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # Возвращаем кортеж из трех обработанных матриц NumPy
        return image_bgr, image_enhanced_gray, image_hsv

    def create_binary_mask(self, image_hsv):
        lower_blue = numpy.array([100, 50, 50])
        upper_blue = numpy.array([140, 255, 255])
        mask_blue_lines = cv2.inRange(image_hsv, lower_blue, upper_blue)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_closed = cv2.morphologyEx(mask_blue_lines, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = numpy.zeros_like(mask_closed)
        cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
        return filled_mask

    def segment_sulfides(self, image_enhanced_gray):
        """
        Сегментирует сульфиды на основе Otsu-порога и фильтрации темных пикселей.
        Используется для необработанных (тестовых) изображений без синих линий разметки.
        """
        # Применяем порог Otsu
        gray = cv2.medianBlur(image_enhanced_gray, 3)
        otsu_value, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        p50, p82, p90, p97 = numpy.percentile(gray, [50, 82, 90, 97])
        contrast = p97 - p50

        if contrast < 35:
            bright_threshold = max(65, min(float(p82), float(otsu_value)))
        else:
            bright_threshold = max(95, min(float(p90), max(float(otsu_value), 115.0)))

        percentile_mask = (gray >= bright_threshold).astype(numpy.uint8) * 255
        min_side = min(gray.shape[:2])
        adaptive_block = min(151, max(31, (min_side // 18) | 1))
        if adaptive_block >= min_side:
            adaptive_block = max(3, min_side - 1)
            if adaptive_block % 2 == 0:
                adaptive_block -= 1
        adaptive_mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            adaptive_block,
            -3
        )
        thresh = cv2.bitwise_or(otsu_mask, percentile_mask)
        if contrast < 55:
            thresh = cv2.bitwise_or(thresh, adaptive_mask)
        
        # Руда/сульфиды должны быть яркими. Отсекаем темную нерудную матрицу.
        # Обычно сульфиды после CLAHE ярче 100-110.
        thresh[gray < 55] = 0
        
        # Убираем мелкие шумы
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small, iterations=1)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        mask_closed = cv2.dilate(mask_closed, kernel_small, iterations=1)
        mask_closed = self.fill_mask_holes(mask_closed, max_hole_area=30000)

        filled = numpy.zeros_like(mask_closed)
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) >= 80:
                cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)

        return self.fill_mask_holes(filled, max_hole_area=50000)

    def remove_small_components(self, mask, min_area=250):
        mask_u8 = (mask > 0).astype(numpy.uint8)
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
        filtered = numpy.zeros_like(mask_u8, dtype=numpy.uint8)
        for label_id in range(1, labels_count):
            if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
                filtered[labels == label_id] = 255
        return filtered

    def fill_mask_holes(self, mask, max_hole_area=20000):
        mask_u8 = (mask > 0).astype(numpy.uint8) * 255
        inv = cv2.bitwise_not(mask_u8)
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats((inv > 0).astype(numpy.uint8), 8)
        filled = mask_u8.copy()
        h, w = mask_u8.shape[:2]
        for label_id in range(1, labels_count):
            x, y, bw, bh, area = stats[label_id]
            touches_border = x == 0 or y == 0 or x + bw >= w or y + bh >= h
            if not touches_border and area <= max_hole_area:
                filled[labels == label_id] = 255
        return filled

    def segment_talc_unet(self, image_bgr, model_path="models/talc.pth", device=None):
        """
        Сегментирует тальк с использованием глубокой сети U-Net (ResNet18).
        Применяет тайлинг (нарезку) для работы с большими разрешениями и сшивает результат.
        """
        import torch
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.unet_model is None:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # Кастомная модель ResNetUNet
                model = ResNetUNet(freeze=False)
                model.load_state_dict(checkpoint["model_state_dict"])
                self.is_custom_unet = True
            else:
                # Стандартный smp.Unet
                try:
                    import segmentation_models_pytorch as smp
                except ImportError:
                    raise ImportError("segmentation_models_pytorch не установлена. Запустите pip install segmentation-models-pytorch")

                model = smp.Unet(
                    encoder_name="resnet18",
                    encoder_weights=None,
                    in_channels=3,
                    classes=1
                )
                model.load_state_dict(checkpoint)
                self.is_custom_unet = False

            model = model.to(device)
            model.eval()
            self.unet_model = model
        else:
            model = self.unet_model

        from preprocessing.tiling import slice_into_tiles, stitch_tiles

        # Применяем нормализацию к изображению перед инференсом:
        # Для ч/б снимков берем оригинал (без CLAHE), для цветных - нормализованный вариант
        is_grayscale = self.is_grayscale_like(image_bgr)
        if is_grayscale:
            image_input = self._ensure_color(image_bgr)
        else:
            image_input = self.normalize_for_inference(image_bgr)

        # Устанавливаем порог адаптивно в зависимости от архитектуры модели:
        if getattr(self, "is_custom_unet", False):
            # Для кастомной модели ResNetUNet (model_boxed.pth) порог 0.5 идеален
            unet_thresh = 0.5
        else:
            # Для стандартного smp.Unet (talc.pth) берем оригинальный высокий порог 0.94
            unet_thresh = 0.94

        tiles, coords = slice_into_tiles(image_input, tile_size=512, overlap=64)

        pred_tiles = []
        with torch.no_grad():
            # Заранее создаем тензоры среднего и отклонения для ImageNet на GPU
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

            for tile in tiles:
                # Переводим в RGB и ресайзим к 256x256 (на которых обучалась сеть)
                tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                tile_resized = cv2.resize(tile_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
                
                # Подготавливаем тензор [1, 3, 256, 256]
                tensor = torch.from_numpy(tile_resized.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
                tensor = tensor.to(device)
                
                # Нормализуем по ImageNet
                tensor = (tensor - mean) / std
                
                # Запуск инференса
                logits = model(tensor)
                if getattr(self, "is_custom_unet", False):
                    # У кастомной ResNetUNet сигмоида уже встроена на выходе
                    probs = logits.cpu().numpy()[0, 0]
                else:
                    # У smp.Unet на выходе логиты
                    probs = torch.sigmoid(logits).cpu().numpy()[0, 0] # [256, 256]
                
                # Порог бинаризации
                pred_tile_resized = (probs > unet_thresh).astype(numpy.uint8) * 255
                
                # Ресайзим обратно к 512x512
                pred_tile = cv2.resize(pred_tile_resized, (512, 512), interpolation=cv2.INTER_NEAREST)
                pred_tiles.append(pred_tile)

        # 3. Сшиваем тайлы обратно
        stitched_mask = stitch_tiles(pred_tiles, coords, image_bgr.shape[:2], tile_size=512)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        stitched_mask = cv2.morphologyEx(stitched_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        stitched_mask = cv2.morphologyEx(stitched_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return self.remove_small_components(stitched_mask, min_area=350)

    def classify_sulfides_resnet(self, image_bgr, sulfide_mask, device=None, model_path="models/ore_resnet18.pth"):
        """
        Классифицирует сульфиды на обычные (класс 1) и тонкие (класс 2) с помощью ResNet-18 скользящим окном.
        """
        import torch
        from torchvision import models, transforms
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.resnet_model is None:
            # Инициализация ResNet-18
            model = models.resnet18()
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, 2)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            self.resnet_model = model
        else:
            model = self.resnet_model

        # Предобработка патчей
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        h, w, _ = image_bgr.shape
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        patch_size = 256
        stride = 128

        y_coords = list(range(0, h - patch_size + 1, stride))
        if y_coords[-1] + patch_size < h:
            y_coords.append(h - patch_size)

        x_coords = list(range(0, w - patch_size + 1, stride))
        if x_coords[-1] + patch_size < w:
            x_coords.append(w - patch_size)

        patches = []
        coords = []

        for y in y_coords:
            for x in x_coords:
                patch = img_rgb[y:y+patch_size, x:x+patch_size]
                patches.append(transform(patch))
                coords.append((y, x))

        # Запуск инференса батчами
        batch_size = 64
        probs = []

        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch = torch.stack(patches[i:i+batch_size]).to(device)
                outputs = model(batch)
                batch_probs = torch.softmax(outputs, dim=1).cpu().numpy()
                probs.extend(batch_probs)

        # Карта вероятностей
        prob_map = numpy.zeros((h, w), dtype=numpy.float32)
        count_map = numpy.zeros((h, w), dtype=numpy.float32)

        for (y, x), prob in zip(coords, probs):
            prob_map[y:y+patch_size, x:x+patch_size] += prob[1]
            count_map[y:y+patch_size, x:x+patch_size] += 1.0

        count_map[count_map == 0] = 1.0
        prob_map /= count_map

        final_mask = numpy.zeros_like(sulfide_mask, dtype=numpy.uint8)
        contours, _ = cv2.findContours(sulfide_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Классификация зерен
        for i, contour in enumerate(contours):
            x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
            if w_c == 0 or h_c == 0:
                continue

            local_mask = numpy.zeros((h_c, w_c), dtype=numpy.uint8)
            shifted_contour = contour - [x_c, y_c]
            cv2.drawContours(local_mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)

            crop_prob = prob_map[y_c:y_c+h_c, x_c:x_c+w_c]
            grain_probs = crop_prob[local_mask == 255]

            if len(grain_probs) == 0:
                continue

            mean_grain_prob = numpy.mean(grain_probs)
            # Класс 1 (Обычные) при prob >= 0.5, иначе Класс 2 (Тонкие)
            val = 1 if mean_grain_prob >= 0.5 else 2

            final_mask_crop = final_mask[y_c:y_c+h_c, x_c:x_c+w_c]
            final_mask_crop[local_mask == 255] = val

        return final_mask

    def classify_ore_segments(self, filled_mask, image_enhanced_gray, brightness_threshold=120, talc_threshold=30, image_bgr=None, use_unet=True, model_path="models/model_boxed.pth", device=None, use_resnet=True, resnet_model_path="models/ore_resnet18.pth"):
        """
        Разделяет изображение на 4 класса:
        - Класс 0 (Порода/Матрица)
        - Класс 1 (Обычные сульфиды / крупная яркая фаза)
        - Класс 2 (Тонкие сульфиды / мелкая/замещенная темная фаза)
        - Класс 3 (Тальк)
        """
        # 1. Проверяем, является ли filled_mask маской разметки (синие линии)
        # Если filled_mask пустая (или почти пустая, менее 1000 пикселей), мы автоматически сегментируем сульфиды
        talc_hint_mask = filled_mask > 0 if numpy.sum(filled_mask > 0) >= 1000 else None
        sulfide_mask = self.segment_sulfides(image_enhanced_gray)

        # 2. Инициализируем итоговую многоклассовую маску
        final_mask = numpy.zeros_like(sulfide_mask, dtype=numpy.uint8)

        # Определяем, является ли изображение фактически ч/б (проходящий свет) или цветным (отраженный)
        # 3. Сегментируем Тальк (Класс 3)
        talc_mask = None
        if use_unet and image_bgr is not None and os.path.exists(model_path):
            try:
                talc_mask = self.segment_talc_unet(image_bgr, model_path, device)
            except Exception as e:
                pass

        if talc_mask is None:
            # Классический фолбек по яркости
            talc_mask = (sulfide_mask == 0) & (image_enhanced_gray < talc_threshold)
        else:
            # U-Net маска, строго ограниченная темной нерудной матрицей (яркость < 97)
            talc_mask = (talc_mask > 127) & (sulfide_mask == 0) & (image_enhanced_gray < 97)

        if talc_hint_mask is not None:
            talc_mask = talc_mask & talc_hint_mask

        talc_mask = self.remove_small_components(talc_mask.astype(numpy.uint8) * 255, min_area=350)
        talc_mask = self.fill_mask_holes(talc_mask, max_hole_area=8000) > 0

        # 4. Классифицируем сульфиды на обычные (Класс 1) и тонкие (Класс 2)
        resnet_success = False
        if use_resnet and image_bgr is not None and os.path.exists(resnet_model_path):
            try:
                resnet_mask = self.classify_sulfides_resnet(image_bgr, sulfide_mask, device, resnet_model_path)
                # Переносим классификацию сульфидов, исключая зоны талька
                final_mask[resnet_mask == 1] = 1
                final_mask[resnet_mask == 2] = 2
                resnet_success = True
            except Exception as e:
                # В случае ошибки падаем на классический пороговый метод
                pass

        if not resnet_success:
            # Классический оптимизированный фолбек по яркости с bounding box (ускорение в ~100 раз)
            contours, _ = cv2.findContours(sulfide_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i, contour in enumerate(contours):
                x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
                if w_c == 0 or h_c == 0:
                    continue

                local_mask = numpy.zeros((h_c, w_c), dtype=numpy.uint8)
                shifted_contour = contour - [x_c, y_c]
                cv2.drawContours(local_mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)

                crop_gray = image_enhanced_gray[y_c:y_c+h_c, x_c:x_c+w_c]
                grain_pixels = crop_gray[local_mask == 255]

                if len(grain_pixels) == 0:
                    continue

                mean_brightness = numpy.mean(grain_pixels)
                val = 1 if mean_brightness >= brightness_threshold else 2

                final_mask_crop = final_mask[y_c:y_c+h_c, x_c:x_c+w_c]
                final_mask_crop[local_mask == 255] = val

        final_mask[(talc_mask) & (final_mask == 0)] = 3

        return final_mask

    def slicer(self, image_path, output_dir):
        image = cv2.imread(image_path)
        output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        if image is None:
            print("Ошибка: Картинка не найдена!")
            return

        height, width, _ = image.shape
        tile_size = 512
        count = 0

        y_cords = list(range(0, height, tile_size))
        if y_cords[-1] + tile_size > height:
            y_cords[-1] = height - tile_size

        x_cords = list(range(0, width, tile_size))
        if x_cords[-1] + tile_size > width:
            x_cords[-1] = width - tile_size

        for y in y_cords:
            for x in x_cords:
                crop = image[y:y + tile_size, x:x + tile_size]
                cv2.imwrite(f"{output_dir}/crop_{count}.jpg", crop)
                count += 1

    def mask_and_image_slicer(self, dirmask, dirimage):
        tile_size = 512
        image_bgr, image_enhanced_gray, image_hsv = self.preprocess_image(dirimage)
        filled_mask = self.create_binary_mask(image_hsv)
        final_mask = self.classify_ore_segments(filled_mask, image_enhanced_gray)
        height, width, _ = image_bgr.shape
        
        y_cords = list(range(0, height, tile_size))
        if y_cords[-1] + tile_size > height:
            y_cords[-1] = height - tile_size

        x_cords = list(range(0, width, tile_size))
        if x_cords[-1] + tile_size > width:
            x_cords[-1] = width - tile_size

        count = 0
        for y in y_cords:
            for x in x_cords:
                crop_img = image_bgr[y: y + tile_size, x: x + tile_size]
                crop_mask_layers = final_mask[y: y + tile_size, x: x + tile_size]

                # Картинку сохраняем как обычно
                img_save_path = os.path.join(dirmask, f"crop_{count}.png")
                cv2.imwrite(img_save_path, crop_img)

                # --- РАСКРАШИВАЕМ 2D МАСКУ В ЦВЕТНУЮ КАРТИНКУ ---
                h_crop, w_crop = crop_mask_layers.shape[:2]
                color_mask = numpy.zeros((h_crop, w_crop, 3), dtype=numpy.uint8)

                # Значение == 1: Руда -> серый (128, 128, 128)
                color_mask[crop_mask_layers == 1] = [128, 128, 128]

                # Значение == 2: Тальк -> белый (255, 255, 255)
                color_mask[crop_mask_layers == 2] = [255, 255, 255]

                # Значение == 3: Границы/срастания -> синий BGR (255, 0, 0)
                color_mask[crop_mask_layers == 3] = [255, 0, 0]

                # Сохраняем маску как нормальный видимый PNG
                mask_save_path = os.path.join(dirmask, f"mask_{count}.png")
                cv2.imwrite(mask_save_path, color_mask)
                count += 1
