# Model Definition, Evaluation & Visualization Module
import torch
import torch.nn as nn


class CTRModel(nn.Module):
    """
    CTR模型
    包含所有模型通用的方法和接口
    """
    def __init__(self, arch, vocab_sizes, num_cols, embedding_dim, hidden_dims, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_fields = len(vocab_sizes)
        self.numerical_dim = len(num_cols) if isinstance(num_cols, (list, tuple)) else int(num_cols)
        
        # SparseEmbedding
        self.sparse_embedding = SparseEmbedding(vocab_sizes, embedding_dim)

        """
        统一的模型构建工厂：根据 arch 返回各种CTR预测模型。
        arch: "deepfm" | "dcn" | "mlp" | "xdeepfm" | "autoint" | "fibi" | "afm" | "nfm"
        """
        if not arch:
            raise ValueError("Architecture cannot be empty")
        
        arch = arch.lower()
        if arch == 'deepfm':
            self.dense = DeepFM_Dense(self.num_fields, self.numerical_dim, embedding_dim, hidden_dims)
        elif arch == 'dcn':
            # allow missing config by providing a sensible default
            num_cross_layers = int(kwargs.get('num_cross_layers', 3))
            self.dense = DCN_Dense(self.numerical_dim, self.num_fields, embedding_dim, hidden_dims, num_cross_layers=num_cross_layers)
        elif arch == 'mlp':
            self.dense = MLP_Dense(self.numerical_dim, self.num_fields, embedding_dim, hidden_dims)
        elif arch == 'xdeepfm':
            # allow missing config by providing a sensible default
            num_cin_layers = int(kwargs.get('num_cin_layers', 3))
            self.dense = xDeepFM_Dense(self.num_fields, self.numerical_dim, embedding_dim, hidden_dims, num_cin_layers=num_cin_layers)
        elif arch == 'autoint':
            # allow missing config by providing sensible defaults
            num_attention_layers = int(kwargs.get('num_attention_layers', 3))
            num_attention_heads = int(kwargs.get('num_attention_heads', 2))
            self.dense = AutoInt_Dense(self.num_fields, self.numerical_dim, embedding_dim, hidden_dims, 
                            num_attention_layers=num_attention_layers, num_attention_heads=num_attention_heads)
        elif arch == 'fibi':
            self.dense = FiBiNET_Dense(self.num_fields, self.numerical_dim, embedding_dim, hidden_dims)
        elif arch == 'afm':
            self.dense = AFM_Dense(self.num_fields, self.numerical_dim, embedding_dim, hidden_dims)
        elif arch == 'nfm':
            self.dense = NFM_Dense(self.numerical_dim, embedding_dim, hidden_dims)
        else:
            raise ValueError(f"Unknown arch: {arch}. Supported: 'deepfm', 'dcn', 'mlp', 'xdeepfm', 'autoint', 'fibi', 'afm', 'nfm'.")
    
    def get_embeddings(self, x_numerical, x_categorical):
        """
        获取拼接后的embeddings（数值特征 + 类别特征embeddings）
        Args:
            x_numerical: 数值特征 [batch_size, numerical_dim]
            x_categorical: 类别特征 [batch_size, num_fields]
        Returns:
            concatenated embeddings [batch_size, numerical_dim + num_fields * embedding_dim]
        """
        cat_embs = self.sparse_embedding(x_categorical)
        cat_emb = cat_embs.view(cat_embs.size(0), -1)
        if isinstance(x_numerical, torch.Tensor) and x_numerical.numel() > 0:
            num = x_numerical
        else:
            num = torch.zeros(x_categorical.size(0), 0, device=cat_emb.device)
        return torch.cat([num, cat_emb], dim=1)
    
    def forward(self, x_numerical, x_categorical):
        # 获取embeddings
        embeddings = self.sparse_embedding(x_categorical)  # [batch_size, num_fields, embedding_dim]
        
        # 通过dense层
        return self.dense(x_numerical, embeddings)


class SparseEmbedding(nn.Module):
    """
    统一的Sparse Embedding类
    将sparse feature映射为embedding向量，输出形状为 [batch_size, num_fields, embedding_dim]
    """
    def __init__(self, vocab_sizes, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_fields = len(vocab_sizes)
        self.vocab_sizes = vocab_sizes
        
        # 为每个field创建embedding层
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_embeddings, embedding_dim) 
            for num_embeddings in vocab_sizes.values()
        ])
    
    def forward(self, x_sparse):
        """
        前向传播
        
        Args:
            x_sparse: sparse特征 [batch_size, num_fields]
        
        Returns:
            embeddings: embedding向量 [batch_size, num_fields, embedding_dim]
        """
        batch_size = x_sparse.size(0)
        embeddings = []
        
        for i, embedding_layer in enumerate(self.embedding_layers):
            field_embedding = embedding_layer(x_sparse[:, i])  # [batch_size, embedding_dim]
            embeddings.append(field_embedding)
        
        # 堆叠为 [batch_size, num_fields, embedding_dim]
        embeddings = torch.stack(embeddings, dim=1)
        return embeddings


class DCN_Dense(nn.Module):
    """DCN模型的dense层参数"""
    def __init__(self, numerical_dim, num_fields, embedding_dim, hidden_dims, num_cross_layers=3):
        super().__init__()
        self.numerical_dim = numerical_dim
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        
        # Input dim = numerical_dim + categorical_emb_dim
        self.input_dim = numerical_dim + num_fields * embedding_dim
        
        # Cross network layers
        self.cross_layers = nn.ModuleList([nn.Linear(self.input_dim, 1, bias=True) for _ in range(num_cross_layers)])
        
        # Deep MLP
        mlp_layers = []
        prev = self.input_dim
        for h in hidden_dims:
            mlp_layers.append(nn.Linear(prev, h))
            mlp_layers.append(nn.LayerNorm(h))
            mlp_layers.append(nn.ReLU())
            prev = h
        self.deep_mlp = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
        
        # Final fusion layer
        fusion_in = self.input_dim + (hidden_dims[-1] if len(hidden_dims) > 0 else self.input_dim)
        self.final_fc = nn.Linear(fusion_in, 1)
    
    def forward(self, x_numerical, embeddings):
        """
        Args:
            x_numerical: 数值特征 [batch_size, numerical_dim]
            embeddings: embedding向量 [batch_size, num_fields, embedding_dim]
        """
        # 构造输入
        if x_numerical.numel() > 0:
            x = torch.cat([x_numerical, embeddings.view(embeddings.size(0), -1)], dim=1)
        else:
            x = embeddings.view(embeddings.size(0), -1)
        
        # Cross part
        x_l = x
        for layer in self.cross_layers:
            alpha = layer(x_l)  # [B, 1]
            x_l = x * alpha + x_l
        x_cross = x_l
        
        # Deep part
        x_deep = self.deep_mlp(x) if not isinstance(self.deep_mlp, nn.Identity) else x
        
        # Fuse
        x_fused = torch.cat([x_cross, x_deep], dim=1)
        return self.final_fc(x_fused)


class DeepFM_Dense(nn.Module):
    """DeepFM模型的dense层参数 - 简化版本，只包含FM二阶部分和Deep部分"""
    def __init__(self, num_fields, numerical_dim, embedding_dim, hidden_dims):
        super().__init__()
        self.num_fields = num_fields
        self.numerical_dim = numerical_dim
        self.embedding_dim = embedding_dim
        
        # 移除所有一阶部分，只保留Deep部分
        
        # Deep part MLP
        deep_in = numerical_dim + self.num_fields * embedding_dim
        layers = []
        prev = deep_in
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev = h
        self.deep_mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.deep_out = nn.Linear(prev, 1) if layers else nn.Linear(deep_in, 1)
    
    def forward(self, x_numerical, embeddings):
        """
        Args:
            x_numerical: 数值特征 [batch_size, numerical_dim]
            embeddings: embedding向量 [batch_size, num_fields, embedding_dim]
        """
        # FM second-order term - 直接使用预计算的embeddings
        sum_v = embeddings.sum(dim=1)
        sum_v_square = sum_v * sum_v
        square_sum_v = (embeddings * embeddings).sum(dim=1)
        fm_second_order = 0.5 * (sum_v_square - square_sum_v).sum(dim=1, keepdim=True)
        
        # Deep part - 直接使用预计算的embeddings
        deep_in = torch.cat([x_numerical, embeddings.view(embeddings.size(0), -1)], dim=1)
        deep_hidden = self.deep_mlp(deep_in) if not isinstance(self.deep_mlp, nn.Identity) else deep_in
        deep_logit = self.deep_out(deep_hidden)
        
        return fm_second_order + deep_logit


class MLP_Dense(nn.Module):
    """MLP模型的dense层参数"""
    def __init__(self, numerical_dim, num_fields, embedding_dim, hidden_dims):
        super().__init__()
        self.numerical_dim = numerical_dim
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        
        # Input dim = numerical_dim + categorical_emb_dim
        input_dim = numerical_dim + num_fields * embedding_dim
        
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x_numerical, embeddings):
        """
        Args:
            x_numerical: 数值特征 [batch_size, numerical_dim]
            embeddings: embedding向量 [batch_size, num_fields, embedding_dim]
        """
        # 构造输入
        if x_numerical.numel() > 0:
            x = torch.cat([x_numerical, embeddings.view(embeddings.size(0), -1)], dim=1)
        else:
            x = embeddings.view(embeddings.size(0), -1)
        
        return self.mlp(x)


class xDeepFM_Dense(nn.Module):
    """xDeepFM模型的dense层参数"""
    def __init__(self, num_fields, numerical_dim, embedding_dim, hidden_dims, num_cin_layers=3):
        super().__init__()
        self.num_fields = num_fields
        self.numerical_dim = numerical_dim
        self.embedding_dim = embedding_dim
        
        # CIN layers
        self.cin_layers = nn.ModuleList()
        field_nums = [num_fields]
        for i in range(num_cin_layers):
            if i == 0:
                cin_layer = nn.Linear(num_fields * num_fields * embedding_dim, 1)
            else:
                cin_layer = nn.Linear(field_nums[-1] * num_fields * embedding_dim, 1)
            self.cin_layers.append(cin_layer)
            field_nums.append(1)
        
        # DNN
        dnn_input_dim = numerical_dim + num_fields * embedding_dim
        dnn_layers = []
        prev = dnn_input_dim
        for h in hidden_dims:
            dnn_layers.append(nn.Linear(prev, h))
            dnn_layers.append(nn.LayerNorm(h))
            dnn_layers.append(nn.ReLU())
            prev = h
        dnn_layers.append(nn.Linear(prev, 1))
        self.dnn = nn.Sequential(*dnn_layers)
        
        # Final linear layer
        self.final_linear = nn.Linear(2, 1)
    
    def forward(self, x_numerical, embeddings):
        """
        Args:
            x_numerical: 数值特征 [batch_size, numerical_dim]
            embeddings: embedding向量 [batch_size, num_fields, embedding_dim]
        """
        # CIN part
        cin_outputs = []
        x0 = embeddings
        xl = x0
        
        for i, cin_layer in enumerate(self.cin_layers):
            xl_outer = torch.einsum('bhd,bmd->bhmd', xl, x0)
            xl_outer = xl_outer.view(xl_outer.size(0), -1)
            xl = cin_layer(xl_outer).unsqueeze(-1)
            cin_outputs.append(xl.squeeze(-1))
        
        cin_output = torch.cat(cin_outputs, dim=1)
        cin_output = cin_output.sum(dim=1, keepdim=True)
        
        # DNN part
        cat_emb = embeddings.view(embeddings.size(0), -1)
        if x_numerical.numel() > 0:
            dnn_input = torch.cat([x_numerical, cat_emb], dim=1)
        else:
            dnn_input = cat_emb
        dnn_output = self.dnn(dnn_input)
        
        # Combine CIN and DNN
        combined = torch.cat([cin_output, dnn_output], dim=1)
        return self.final_linear(combined)


class AutoInt_Dense(nn.Module):
    """AutoInt模型的dense层参数"""
    def __init__(self, num_fields, numerical_dim, embedding_dim, hidden_dims, num_attention_layers=3, num_attention_heads=2):
        super().__init__()
        self.num_fields = num_fields
        self.numerical_dim = numerical_dim
        self.embedding_dim = embedding_dim
        
        # Self-attention layers
        self.attention_layers = nn.ModuleList()
        for _ in range(num_attention_layers):
            attention_layer = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=num_attention_heads,
                batch_first=True
            )
            self.attention_layers.append(attention_layer)
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(num_attention_layers)
        ])
        
        # MLP
        mlp_input_dim = numerical_dim + num_fields * embedding_dim
        mlp_layers = []
        prev = mlp_input_dim
        for h in hidden_dims:
            mlp_layers.append(nn.Linear(prev, h))
            mlp_layers.append(nn.LayerNorm(h))
            mlp_layers.append(nn.ReLU())
            prev = h
        mlp_layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, x_numerical, embeddings):
        """
        Args:
            x_numerical: 数值特征 [batch_size, numerical_dim]
            embeddings: embedding向量 [batch_size, num_fields, embedding_dim]
        """
        # Self-attention layers
        attended_embeddings = embeddings
        for attention_layer, layer_norm in zip(self.attention_layers, self.layer_norms):
            attended, _ = attention_layer(attended_embeddings, attended_embeddings, attended_embeddings)
            attended_embeddings = layer_norm(attended + attended_embeddings)
        
        # Flatten for MLP
        cat_emb = attended_embeddings.view(attended_embeddings.size(0), -1)
        if x_numerical.numel() > 0:
            mlp_input = torch.cat([x_numerical, cat_emb], dim=1)
        else:
            mlp_input = cat_emb
        return self.mlp(mlp_input)


class FiBiNET_Dense(nn.Module):
    """FiBiNET模型的dense层参数"""
    def __init__(self, num_fields, numerical_dim, embedding_dim, hidden_dims):
        super().__init__()
        self.num_fields = num_fields
        self.numerical_dim = numerical_dim
        self.embedding_dim = embedding_dim
        
        # SENET layers
        self.senet = SENETLayer(num_fields, reduction_ratio=3)
        
        # Bilinear interaction layers
        self.bilinear_interaction = BilinearInteractionLayer(num_fields, embedding_dim)
        
        # MLP
        bilinear_output_dim = num_fields * (num_fields - 1) // 2 * embedding_dim
        mlp_input_dim = numerical_dim + num_fields * embedding_dim + bilinear_output_dim
        mlp_layers = []
        prev = mlp_input_dim
        for h in hidden_dims:
            mlp_layers.append(nn.Linear(prev, h))
            mlp_layers.append(nn.LayerNorm(h))
            mlp_layers.append(nn.ReLU())
            prev = h
        mlp_layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, x_numerical, embeddings):
        """
        Args:
            x_numerical: 数值特征 [batch_size, numerical_dim]
            embeddings: embedding向量 [batch_size, num_fields, embedding_dim]
        """
        # SENET
        senet_embeddings = self.senet(embeddings)
        
        # Bilinear interaction
        bilinear_output = self.bilinear_interaction(senet_embeddings)
        
        # Original embeddings
        original_emb = embeddings.view(embeddings.size(0), -1)
        
        # Combine
        if x_numerical.numel() > 0:
            mlp_input = torch.cat([x_numerical, original_emb, bilinear_output], dim=1)
        else:
            mlp_input = torch.cat([original_emb, bilinear_output], dim=1)
        return self.mlp(mlp_input)


class AFM_Dense(nn.Module):
    """AFM模型的dense层参数"""
    def __init__(self, num_fields, numerical_dim, embedding_dim, hidden_dims, attention_dim=8):
        super().__init__()
        self.num_fields = num_fields
        self.numerical_dim = numerical_dim
        self.embedding_dim = embedding_dim
        
        # Attention network
        self.attention_linear = nn.Linear(embedding_dim, attention_dim)
        self.attention_h = nn.Linear(attention_dim, 1, bias=False)
        
        # MLP
        mlp_input_dim = numerical_dim + num_fields * embedding_dim
        mlp_layers = []
        prev = mlp_input_dim
        for h in hidden_dims:
            mlp_layers.append(nn.Linear(prev, h))
            mlp_layers.append(nn.LayerNorm(h))
            mlp_layers.append(nn.ReLU())
            prev = h
        mlp_layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, x_numerical, embeddings):
        """
        Args:
            x_numerical: 数值特征 [batch_size, numerical_dim]
            embeddings: embedding向量 [batch_size, num_fields, embedding_dim]
        """
        # Element-wise product for feature interactions
        element_wise_products = []
        for i in range(self.num_fields):
            for j in range(i + 1, self.num_fields):
                element_wise_products.append(embeddings[:, i] * embeddings[:, j])
        
        element_wise_products = torch.stack(element_wise_products, dim=1)
        
        # Attention mechanism
        attention_scores = torch.relu(self.attention_linear(element_wise_products))
        attention_scores = self.attention_h(attention_scores).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Weighted sum
        attended_interactions = element_wise_products * attention_weights.unsqueeze(-1)
        attended_output = attended_interactions.sum(dim=1)
        
        # MLP part
        cat_emb = embeddings.view(embeddings.size(0), -1)
        if x_numerical.numel() > 0:
            mlp_input = torch.cat([x_numerical, cat_emb], dim=1)
        else:
            mlp_input = cat_emb
        mlp_output = self.mlp(mlp_input)
        
        # Combine
        interaction_output = attended_output.sum(dim=1, keepdim=True)
        return mlp_output + interaction_output


class NFM_Dense(nn.Module):
    """NFM模型的dense层参数"""
    def __init__(self, numerical_dim, embedding_dim, hidden_dims):
        super().__init__()
        self.numerical_dim = numerical_dim
        self.embedding_dim = embedding_dim
        
        # Bi-interaction pooling
        self.bi_interaction = BiInteractionPooling()
        
        # MLP
        mlp_input_dim = numerical_dim + embedding_dim
        mlp_layers = []
        prev = mlp_input_dim
        for h in hidden_dims:
            mlp_layers.append(nn.Linear(prev, h))
            mlp_layers.append(nn.LayerNorm(h))
            mlp_layers.append(nn.ReLU())
            prev = h
        mlp_layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, x_numerical, embeddings):
        """
        Args:
            x_numerical: 数值特征 [batch_size, numerical_dim]
            embeddings: embedding向量 [batch_size, num_fields, embedding_dim]
        """
        # Bi-interaction pooling
        bi_interaction_output = self.bi_interaction(embeddings)
        
        # MLP
        if x_numerical.numel() > 0:
            mlp_input = torch.cat([x_numerical, bi_interaction_output], dim=1)
        else:
            mlp_input = bi_interaction_output
        return self.mlp(mlp_input)



class SENETLayer(nn.Module):
    """SENET: Squeeze-and-Excitation Network"""
    def __init__(self, num_fields, reduction_ratio=3):
        super().__init__()
        self.num_fields = num_fields
        self.reduction_ratio = reduction_ratio
        self.reduced_size = max(1, num_fields // reduction_ratio)
        
        self.squeeze = nn.Linear(num_fields, self.reduced_size)
        self.excitation = nn.Linear(self.reduced_size, num_fields)
        
    def forward(self, inputs):
        # Squeeze: Global average pooling
        Z = inputs.mean(dim=2)  # [batch, num_fields]
        
        # Excitation
        A = torch.sigmoid(self.excitation(torch.relu(self.squeeze(Z))))  # [batch, num_fields]
        
        # Scale
        V = inputs * A.unsqueeze(2)  # [batch, num_fields, embedding_dim]
        return V


class BilinearInteractionLayer(nn.Module):
    """Bilinear Interaction Layer"""
    def __init__(self, num_fields, embedding_dim, bilinear_type='field_all'):
        super().__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.bilinear_type = bilinear_type
        
        if bilinear_type == 'field_all':
            self.bilinear = nn.Linear(embedding_dim, embedding_dim, bias=False)
        elif bilinear_type == 'field_each':
            self.bilinear = nn.ModuleList([
                nn.Linear(embedding_dim, embedding_dim, bias=False) for _ in range(num_fields)
            ])
        elif bilinear_type == 'field_interaction':
            self.bilinear = nn.ModuleList([
                nn.Linear(embedding_dim, embedding_dim, bias=False) 
                for _ in range(num_fields * (num_fields - 1) // 2)
            ])
    
    def forward(self, inputs):
        # inputs: [batch, num_fields, embedding_dim]
        if self.bilinear_type == 'field_all':
            p = [self.bilinear(inputs[:, i]) * inputs[:, j] 
                 for i in range(self.num_fields) for j in range(i + 1, self.num_fields)]
        elif self.bilinear_type == 'field_each':
            p = [self.bilinear[i](inputs[:, i]) * inputs[:, j] 
                 for i in range(self.num_fields) for j in range(i + 1, self.num_fields)]
        else:  # field_interaction
            p = []
            idx = 0
            for i in range(self.num_fields):
                for j in range(i + 1, self.num_fields):
                    p.append(self.bilinear[idx](inputs[:, i]) * inputs[:, j])
                    idx += 1
        
        return torch.cat(p, dim=1)  # [batch, num_fields * (num_fields - 1) // 2 * embedding_dim]


class BiInteractionPooling(nn.Module):
    """Bi-interaction Pooling Layer"""
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        # inputs: [batch, num_fields, embedding_dim]
        # Sum of squares
        sum_of_squares = inputs.sum(dim=1) ** 2  # [batch, embedding_dim]
        # Square of sum
        square_of_sum = (inputs ** 2).sum(dim=1)  # [batch, embedding_dim]
        # Bi-interaction
        bi_interaction = 0.5 * (sum_of_squares - square_of_sum)  # [batch, embedding_dim]
        return bi_interaction

