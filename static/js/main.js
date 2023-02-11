$(function(){
	$('#gaussion').click(function(){
		var kata = $('#kata').val();
		$.ajax({
			url: '/gaussion',
			data: $('form').serialize(),
			type: 'POST',
			beforeSend: function(){
        $('.loading').addClass('loading-show');
			},
			success: function(response){
        $('.algoritma').text('Gaussian Naive Bayes');
        $('.result').text(response.hate_speech);
        $('.accuracy').text(response.accuracy);
			},
			error: function(error){
				console.log(error);
			},
			complete: function(){
				$('.loading').removeClass('loading-show');
		
			},
		});
	});

  $('#gaussian_bernoulli').click(function(){
		var kata = $('#kata').val();
		$.ajax({
			url: '/gausion_bernolui',
			data: $('form').serialize(),
			type: 'POST',
			beforeSend: function(){
        $('.loading').addClass('loading-show');
			},
			success: function(response){
        $('.algoritma').text('Gaussian Bernouli');
        $('.result').text(response.hate_speech);
        $('.accuracy').text(response.accuracy);
			},
			error: function(error){
				console.log(error);
			},
			complete: function(){
				$('.loading').removeClass('loading-show');
		
			},
		});
	});

  $('#gaussian_multinominal').click(function(){
		var kata = $('#kata').val();
		$.ajax({
			url: '/gaussian_multinominal',
			data: $('form').serialize(),
			type: 'POST',
			beforeSend: function(){
        $('.loading').addClass('loading-show');
			},
			success: function(response){
        $('.algoritma').text('Gaussian Multinominal ');
        $('.result').text(response.hate_speech);
        $('.accuracy').text(response.accuracy);
			},
			error: function(error){
				console.log(error);
			},
			complete: function(){
				$('.loading').removeClass('loading-show');
		
			},
		});
	});





	
})